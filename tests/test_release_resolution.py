import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from release_fixtures import admission as admission
from release_fixtures import arguments as arguments
from release_fixtures import benchmark as benchmark
from release_fixtures import records as records
from release_fixtures import setup_executor as setup_executor
from sqlalchemy.orm import Session
from test_release_catalog import releases as releases

from cannery.artifacts import ArtifactError
from cannery.registry.models import SpecVersion
from cannery.releases import state
from cannery.releases.contract import read_manifest
from cannery.releases.download_api import install_downloads
from cannery.releases.eligibility import publication_manifest
from cannery.releases.models import Publication
from cannery.releases.worker import process
from cannery.settings import Settings

ACTOR = {"namespace": "team", "role": "user"}


@pytest.fixture
def published(admission):
    engine, key, store, _ = admission
    m = publication_manifest(engine, key)
    process(engine, state.request_publication(engine, key, ACTOR), store)
    with Session(engine) as db:
        release_id = db.get(Publication, key).release_id
    app = FastAPI()
    app.state.artifacts = store
    install_downloads(app, Settings(dev_identity_enabled=True, release_enabled=True), engine)
    return engine, key, store, m, release_id, TestClient(app)


def body(manifest, **changes):
    return {
        "kernel": manifest.kernel,
        "runtime": manifest.cohort.runtime,
        "shape": [2, 3],
    } | changes


def test_resolve_downloads_pinned_package_without_backend_credentials(published):
    _, _, store, m, identifier, client = published
    response = client.post("/api/v1/artifacts/resolve", json=body(m))
    assert response.status_code == 200
    assert response.json()["id"] == str(identifier)
    pin = response.json()["manifest_sha256"]
    root = f"/api/v1/releases/{identifier}"
    manifest = client.get(root + "/manifest")
    assert manifest.status_code == 200
    saved = read_manifest(manifest.content, pin)
    package = client.get(root + "/package")
    assert package.status_code == 200
    assert package.content == store.get(saved.package) and saved.package.sha256 == m.package.sha256
    assert package.headers["cache-control"] == "private, no-store"
    assert not any(s in response.text for s in ["write_token", "read_token", "artifacts:8000"])


def test_pinned_older_release_never_silently_changes_specification(published):
    engine, _, _, m, identifier, client = published
    with Session(engine) as db, db.begin():
        frozen = db.get(SpecVersion, m.spec_id)
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
    assert client.post("/api/v1/artifacts/resolve", json=body(m)).status_code == 404
    selected = client.post("/api/v1/artifacts/resolve", json=body(m, release_id=str(identifier)))
    assert selected.status_code == 200 and selected.json()["spec_version"] == 1
    assert (
        client.post(
            "/api/v1/artifacts/resolve", json=body(m, release_id=str(identifier), spec_version=2)
        ).status_code
        == 409
    )
    assert (
        client.post(
            "/api/v1/artifacts/resolve",
            json=body(m, release_id=str(identifier), runtime=m.cohort.runtime | {"cann": "other"}),
        ).status_code
        == 409
    )


@pytest.mark.parametrize(
    "change",
    [
        {"shape": [True, 3]},
        {"shape": [-1, 3]},
        {"dtype": "float16"},
        {"rtol": 1.0},
        {"runtime": {"machine": "aarch64"}},
        {"force": True},
    ],
)
def test_invalid_or_unsupported_resolution_inputs_are_rejected(published, change):
    _, _, _, m, _, client = published
    assert client.post("/api/v1/artifacts/resolve", json=body(m, **change)).status_code == 422


def test_unauthorized_consumers_cannot_fetch_private_blobs(published):
    engine, _, store, m, identifier, _ = published
    app = FastAPI()
    app.state.artifacts = store
    install_downloads(app, Settings(release_enabled=True), engine)
    client = TestClient(app)
    assert client.post("/api/v1/artifacts/resolve", json=body(m)).status_code == 401
    assert client.get(f"/api/v1/releases/{identifier}/manifest").status_code == 401
    assert client.get(f"/api/v1/releases/{identifier}/package").status_code == 401


def test_withdrawal_blocks_resolution_and_downloads_without_deleting_bytes(published):
    engine, key, store, m, identifier, client = published
    before = dict(store.data)
    assert state.withdraw(engine, key, ACTOR)
    assert client.post("/api/v1/artifacts/resolve", json=body(m)).status_code == 404
    assert client.get(f"/api/v1/releases/{identifier}/package").status_code == 404
    assert client.get(f"/api/v1/releases/{identifier}/manifest").status_code == 404
    assert store.data == before


def test_storage_outage_and_corruption_never_send_unverified_bytes(published, monkeypatch):
    _, _, store, m, identifier, client = published
    original = store.get

    def offline(_):
        raise ArtifactError("offline")

    monkeypatch.setattr(store, "get", offline)
    assert client.get(f"/api/v1/releases/{identifier}/package").status_code == 503
    monkeypatch.setattr(store, "get", original)
    store.data[m.package.sha256] = b"bad"
    assert client.get(f"/api/v1/releases/{identifier}/package").status_code == 503


def test_withdrawal_during_backend_read_is_checked_before_response(published, monkeypatch):
    engine, key, store, _, identifier, client = published
    original = store.get

    def delayed(ref):
        raw = original(ref)
        state.withdraw(engine, key, ACTOR)
        return raw

    monkeypatch.setattr(store, "get", delayed)
    assert client.get(f"/api/v1/releases/{identifier}/package").status_code == 404


def test_catalog_lists_scoreless_releases_for_explicit_consumption(releases):
    _, add, client, m = releases
    scored = add(0.5)
    empty = add(0.5, empty=True)
    reply = client.get("/api/v1/releases", params={"kernel": m.kernel, "limit": 1, "offset": 1})
    assert reply.status_code == 200
    assert reply.json()["total"] == 2
    assert reply.json()["rows"][0]["id"] == str(empty[0])
    assert reply.json()["rows"][0]["score"]["geomean_speedup"] is None
    assert client.get("/api/v1/leaderboard", params={"kernel": m.kernel}).json()["rows"][0][
        "id"
    ] == str(scored[0])


def test_resolution_checks_language_static_shapes_and_runtime(releases):
    from cannery.releases.catalog import SelectionError
    from cannery.releases.resolution import ResolveRequest, resolve

    engine, add, _, m = releases
    native = add(0.5)
    catlass = add(0.7, "catlass")
    add(0.8, driver="other")
    with Session(engine) as db:
        # A second driver cohort remains ambiguous even with the same runtime.
        with pytest.raises(SelectionError, match="cohort_required"):
            resolve(db, ResolveRequest(**body(m)))
        assert resolve(db, ResolveRequest(**body(m, cohort=native[2])))["id"] == str(catlass[0])
        assert resolve(db, ResolveRequest(**body(m, cohort=native[2], shape=[999])))["id"] == str(
            native[0]
        )
        assert resolve(db, ResolveRequest(**body(m, cohort=native[2], language="ascendc")))[
            "id"
        ] == str(native[0])
        with pytest.raises(SelectionError):
            resolve(db, ResolveRequest(**body(m, runtime=m.cohort.runtime | {"cann": "other"})))
