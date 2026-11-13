from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from release_fixtures import admission as admission
from release_fixtures import arguments as arguments
from release_fixtures import benchmark as benchmark
from release_fixtures import records as records
from release_fixtures import setup_executor as setup_executor
from sqlalchemy.orm import Session
from test_build_state import candidate

from cannery.artifacts import digest
from cannery.registry.models import SpecVersion
from cannery.releases.contract import ReleaseManifest
from cannery.releases.discovery_api import install_discovery
from cannery.releases.eligibility import publication_manifest
from cannery.releases.models import Publication, Release, ReleaseTolerance
from cannery.settings import Settings


@pytest.fixture
def releases(admission, records):
    # Synthetic immutable releases isolate ranking policy from runtime execution.
    engine, key, _, _ = admission
    template = publication_manifest(engine, key)
    serial = [0]

    def add(
        ratio, language="ascendc", driver=None, empty=False, extra_tolerance=False, device="a5"
    ):
        identifier = candidate(records)
        data = template.model_dump(mode="json")
        data.update(
            submission_id=str(identifier),
            language=language,
            abi="unary-f32-v1" if language == "ascendc" else "catlass-unary-f32-v1",
        )
        for case in data["score"]["cases"]:
            if case["excluded"]:
                continue
            case["reference"] = {"median_us": 20.0, "p95_us": 22.0, "samples": 100}
            case["candidate"] = {"median_us": 20.0 / ratio, "p95_us": 22.0 / ratio, "samples": 100}
            case["speedup"] = ratio
        data["score"]["geomean_speedup"] = ratio
        data["device"] = data["cohort"]["device"] = device
        if device == "a3":
            data["cohort"]["runtime"]["soc"] = "Ascend910B4-1"
        if driver:
            data["cohort"]["driver"] = driver
        if empty:
            data["score"].update(cases=[data["score"]["cases"][-1]], geomean_speedup=None)
        if extra_tolerance:
            data["tolerances"] = [data["tolerances"][0], {"rtol": 0.0, "atol": 0.0}]
        if language == "catlass":
            data["specializations"] = [
                {"shape": c["shape"], "dtype": "float32"} for c in data["score"]["cases"]
            ]
        import json

        m = ReleaseManifest.model_validate_json(json.dumps(data))
        sha = digest(m.encoded())
        release_id = uuid4()
        row = Release(
            id=release_id,
            document=m.model_dump(mode="json"),
            manifest=dict(repository="private", revision=sha, path="manifest.json", sha256=sha),
        )
        with Session(engine) as db, db.begin():
            db.add(row)
            db.flush()
            for t in m.tolerances:
                db.add(ReleaseTolerance(release_id=row.id, rtol=t.rtol, atol=t.atol))
            serial[0] += 1
            db.add(
                Publication(
                    candidate_id=identifier,
                    release_id=row.id,
                    status="published",
                    generation=1,
                    actor={"namespace": "team", "role": "user"},
                    published_at=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=serial[0]),
                )
            )
        return release_id, identifier, m.cohort_sha256

    app = FastAPI()
    install_discovery(app, Settings(release_enabled=True), engine)
    return engine, add, TestClient(app), template


def test_private_results_are_absent_and_slow_published_results_rank(releases):
    engine, add, client, template = releases
    assert client.get("/api/v1/releases/kernels").json() == []
    first = add(0.5)
    second = add(0.7, "catlass")
    add(0.6)
    response = client.get("/api/v1/leaderboard", params={"kernel": template.kernel})
    assert response.status_code == 200
    body = response.json()
    assert [r["score"]["geomean_speedup"] for r in body["rows"]] == [0.7, 0.6, 0.5]
    assert body["rows"][0]["id"] == str(second[0]) and body["total"] == 3
    assert client.get("/api/v1/releases/kernels").json() == [
        {"kernel": template.kernel, "versions": [1], "devices": ["a5"]}
    ]
    points = client.get("/api/v1/leaderboard/history", params={"kernel": template.kernel}).json()[
        "points"
    ]
    assert [(p["geomean_speedup"], p["language"]) for p in points] == [
        (0.5, "ascendc"),
        (0.7, "catlass"),
    ]
    info = client.get("/api/v1/releases/" + str(first[0])).json()
    assert info["source"] == template.source.model_dump(mode="json") and "worker" not in str(info)


def test_withdrawal_removes_current_ranking_history_and_metadata(releases):
    engine, add, client, template = releases
    a = add(0.5)
    b = add(0.7)
    with Session(engine) as db, db.begin():
        db.get(Publication, b[1]).status = "withdrawn"
    rows = client.get("/api/v1/leaderboard", params={"kernel": template.kernel}).json()["rows"]
    assert [r["id"] for r in rows] == [str(a[0])]
    assert client.get("/api/v1/releases/" + str(b[0])).status_code == 404
    assert (
        len(
            client.get("/api/v1/leaderboard/history", params={"kernel": template.kernel}).json()[
                "points"
            ]
        )
        == 1
    )


def test_cohorts_never_mix_and_unknown_tolerances_never_infer_a_pass(releases):
    _, add, client, template = releases
    a = add(0.5, extra_tolerance=True)
    b = add(2.0, driver="other")
    assert client.get("/api/v1/leaderboard", params={"kernel": template.kernel}).status_code == 409
    params = {"kernel": template.kernel, "cohort": a[2]}
    body = client.get("/api/v1/leaderboard", params=params | {"rtol": 0, "atol": 0}).json()
    assert body["total"] == 1 and body["rows"][0]["id"] == str(a[0])
    assert (
        client.get("/api/v1/leaderboard", params=params | {"rtol": 1.0, "atol": 1.0}).status_code
        == 422
    )
    assert client.get("/api/v1/leaderboard", params=params | {"rtol": 0}).status_code == 422
    assert (
        client.get(
            "/api/v1/leaderboard", params=params | {"tolerance_profile": "loose"}
        ).status_code
        == 422
    )
    assert (
        client.get(
            "/api/v1/leaderboard", params={"kernel": template.kernel, "cohort": b[2]}
        ).json()["total"]
        == 1
    )


def test_latest_spec_does_not_silently_fall_back_and_empty_releases_are_unranked(releases):
    engine, add, client, template = releases
    add(0.5)
    add(1.0, empty=True)
    assert (
        client.get("/api/v1/leaderboard", params={"kernel": template.kernel}).json()["total"] == 1
    )
    with Session(engine) as db, db.begin():
        frozen = db.get(SpecVersion, template.spec_id)
        db.add(
            SpecVersion(
                kernel_id=frozen.kernel_id,
                version=2,
                content_sha256="f" * 64,
                manifest=frozen.manifest,
                snapshot={},
                file_sha256={},
                smoke={},
            )
        )
    assert client.get("/api/v1/leaderboard", params={"kernel": template.kernel}).status_code == 404
    assert (
        client.get(
            "/api/v1/leaderboard", params={"kernel": template.kernel, "spec_version": 1}
        ).json()["total"]
        == 1
    )


def test_ties_are_stable_and_never_create_an_extra_record_point(releases):
    engine, add, client, template = releases
    first, second = add(0.5), add(0.5)
    with Session(engine) as db, db.begin():
        db.get(Publication, second[1]).published_at = db.get(Publication, first[1]).published_at
    params = {"kernel": template.kernel}
    rows = client.get("/api/v1/leaderboard", params=params).json()["rows"]
    assert [r["id"] for r in rows] == sorted([str(first[0]), str(second[0])])
    points = client.get("/api/v1/leaderboard/history", params=params).json()["points"]
    assert len(points) == 1 and points[0]["release_id"] == rows[0]["id"]
    page = client.get("/api/v1/leaderboard", params=params | {"limit": 1, "offset": 1}).json()
    assert page["total"] == 2 and page["rows"] == rows[1:]


def test_a3_a5_rankings_histories_and_cohorts_are_disjoint(releases):
    _, add, client, template = releases
    a5 = add(0.7)
    a3 = add(1.5, device="a3")
    for device, item in [("a5", a5), ("a3", a3)]:
        params = {"kernel": template.kernel, "device": device}
        ranking = client.get("/api/v1/leaderboard", params=params)
        assert ranking.status_code == 200
        assert [r["id"] for r in ranking.json()["rows"]] == [str(item[0])]
        assert ranking.json()["cohorts"] == [item[2]]
        points = client.get("/api/v1/leaderboard/history", params=params).json()["points"]
        assert len(points) == 1 and points[0]["geomean_speedup"] == (0.7 if device == "a5" else 1.5)
        wrong = client.get(
            "/api/v1/leaderboard", params=params | {"cohort": a3[2] if device == "a5" else a5[2]}
        )
        assert wrong.status_code == 404
