import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from release_fixtures import admission as admission
from release_fixtures import arguments as arguments
from release_fixtures import benchmark as benchmark
from release_fixtures import records as records
from release_fixtures import setup_executor as setup_executor
from test_release_catalog import releases as releases

from cannery.releases import state
from cannery.releases.worker import process
from cannery.settings import Settings
from cannery.web import install_web
from cannery.web.releases import chart


def web(engine):
    app = FastAPI()
    install_web(app, Settings(release_enabled=True), engine)
    return TestClient(app)


def test_global_best_empty_partition_chart_and_withdrawal(releases):
    engine, add, _, m = releases
    client = web(engine)
    assert "No published results yet" in client.get("/global-best").text
    first, second = add(0.5), add(0.7, "catlass")
    params = {"kernel": m.kernel, "spec_version": "", "cohort": "", "rtol": "", "atol": ""}
    page = client.get("/global-best", params=params)
    assert page.status_code == 200
    assert "One strict best-so-far curve" in page.text
    assert str(first[0]) in page.text and str(second[0]) in page.text
    assert "0.7×" in page.text and "CATLASS" in page.text
    state.withdraw(engine, second[1], {"namespace": "team", "role": "user"})
    assert str(second[0]) not in client.get("/global-best", params=params).text
    assert client.get(f"/releases/{second[0]}").status_code == 404
    add(0.8, driver="other")
    assert "cohort_required" in client.get("/global-best", params=params).text
    assert (
        client.get(
            "/global-best", params={"kernel": m.kernel, "rtol": "nan", "atol": "0"}
        ).status_code
        == 422
    )


def test_detail_includes_saved_numerical_metrics_without_private_evidence(admission):
    engine, key, store, _ = admission
    process(
        engine, state.request_publication(engine, key, {"namespace": "team", "role": "user"}), store
    )
    from sqlalchemy.orm import Session

    from cannery.releases.models import Publication

    with Session(engine) as db:
        identifier = db.get(Publication, key).release_id
    page = web(engine).get(f"/releases/{identifier}")
    assert page.status_code == 200
    assert (
        "Saved numerical validation" in page.text
        and "Maximum absolute / relative error" in page.text
    )
    assert "private_evidence" not in page.text and "write_token" not in page.text
    assert "Manifest SHA256" in page.text and "Package SHA256" in page.text


def test_chart_uses_publication_times_and_includes_reference_line():
    points = [
        {
            "published_at": f"2026-01-01T00:00:{n:02d}+00:00",
            "geomean_speedup": v,
            "language": "ascendc",
        }
        for n, v in [(0, 0.5), (2, 0.7), (10, 1.2)]
    ]
    value = chart(points)
    assert value["points"][1]["x"] == pytest.approx(190)
    assert value["points"][2]["x"] == pytest.approx(750)
    assert value["line"].count(" H ") == 2 and value["line"].count(" V ") == 2
    assert 0 < value["reference_y"] < 220
    assert chart([])["points"] == []
