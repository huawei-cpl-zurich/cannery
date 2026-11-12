from uuid import UUID

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from release_fixtures import admission as admission
from release_fixtures import arguments as arguments
from release_fixtures import benchmark as benchmark
from release_fixtures import records as records
from release_fixtures import setup_executor as setup_executor
from sqlalchemy import update
from sqlalchemy.orm import Session

from cannery.benchmark.state import PerformanceJob
from cannery.releases.api import install_publications
from cannery.releases.models import Publication, PublicationJob
from cannery.releases.worker import process
from cannery.settings import Settings
from cannery.submissions.api import get_info, install_submissions


def client(engine, **changes):
    settings = Settings(dev_identity_enabled=True, release_enabled=True, **changes)
    app = FastAPI()
    install_publications(app, settings, engine)
    install_submissions(app, settings, engine)
    return TestClient(app)


def test_explicit_publication_status_and_withdrawal(admission):
    engine, key, store, _ = admission
    c = client(engine)
    assert c.get(f"/api/v1/submissions/{key}").json()["publication"]["status"] == "private"
    route = f"/api/v1/submissions/{key}"
    response = c.post(route + "/publish", json={})
    assert response.status_code == 202
    assert c.post(route + "/publish", json={}).json() == response.json()
    job = UUID(response.json()["id"])
    assert c.get(response.json()["status_url"]).json()["status"] == "queued"
    assert c.get(route + "/publication").json()["status"] == "publishing"
    process(engine, job, store)
    info = c.get(route).json()
    assert info["status"] == "complete"
    assert info["publication"]["status"] == "published"
    release = info["publication"]["release_id"]
    assert c.post(route + "/withdraw", json={}).json()["status"] == "withdrawn"
    assert c.post(route + "/withdraw", json={}).json()["status"] == "withdrawn"
    second = UUID(c.post(route + "/publish", json={}).json()["id"])
    process(engine, second, store)
    assert c.get(route + "/publication").json()["release_id"] == release


@pytest.mark.parametrize(
    "identity,role,code", [(False, "user", 401), (True, "user", 404), (True, "admin", 202)]
)
def test_only_owner_or_authorized_admin_can_publish(admission, identity, role, code):
    engine, key, _, _ = admission
    settings = Settings(
        dev_identity_enabled=identity, dev_namespace="other", dev_role=role, release_enabled=True
    )
    app = FastAPI()
    install_publications(app, settings, engine)
    c = TestClient(app)
    r = c.post(f"/api/v1/submissions/{key}/publish", json={})
    assert r.status_code == code
    with Session(engine) as db:
        pub = db.get(Publication, key)
        assert (pub is not None) == (code == 202)
        if pub:
            assert pub.actor == {"namespace": "other", "role": "admin"}


def test_failed_validation_and_forged_action_body_never_queue(admission):
    engine, key, _, _ = admission
    c = client(engine)
    route = f"/api/v1/submissions/{key}"
    assert c.post(route + "/publish", json={"force": True}).status_code == 422
    assert c.post(route + "/publish", data={}).status_code == 422
    with engine.begin() as conn:
        conn.execute(
            update(PerformanceJob).where(PerformanceJob.candidate_id == key).values(status="failed")
        )
    assert c.post(route + "/publish", json={}).status_code == 409
    with Session(engine) as db:
        assert db.query(PublicationJob).count() == 0


def test_publication_jobs_and_status_remain_namespace_owned(admission):
    engine, key, _, _ = admission
    owner = client(engine)
    job = owner.post(f"/api/v1/submissions/{key}/publish", json={}).json()
    other = client(engine, dev_namespace="other")
    assert other.get(job["status_url"]).status_code == 404
    assert other.get(job["publication_url"]).status_code == 404
    assert other.post(f"/api/v1/submissions/{key}/withdraw", json={}).status_code == 404


def test_disabled_feature_preserves_old_submission_contract(admission):
    engine, key, _, _ = admission
    app = FastAPI()
    settings = Settings(dev_identity_enabled=True)
    install_publications(app, settings, engine)
    c = TestClient(app)
    assert c.post(f"/api/v1/submissions/{key}/publish", json={}).status_code == 404
    with Session(engine) as db:
        assert "publication" not in get_info(db, "team", key)


def test_release_feature_rejects_nonprivate_storage_before_startup():
    from cannery.api import create_app

    with pytest.raises(ValueError, match="private artifact"):
        create_app(Settings(release_enabled=True, artifact_backend="fake"))
