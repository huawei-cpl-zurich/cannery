from unittest.mock import MagicMock
from uuid import UUID

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError
from sqlalchemy.orm import Session
from test_registration_api import client as client

from cannery.registry.freeze import kernel_key
from cannery.registry.models import Kernel, SpecVersion
from cannery.submissions.models import Candidate, Submission
from cannery.submissions.state import IntakeState


@pytest.fixture(autouse=True)
def dispatch_mock(monkeypatch):
    monkeypatch.setattr("cannery.submissions.api.publish_submission", MagicMock())


def seed(client):
    engine = client.app.state.test_engine
    with Session(engine) as session, session.begin():
        key = kernel_key("team", "identity")
        session.add(Kernel(id=key, namespace="team", name="identity"))
        session.flush()
        spec = SpecVersion(
            kernel_id=key,
            version=1,
            content_sha256="a" * 64,
            manifest={"kernel": {"namespace": "team", "name": "identity"}},
            snapshot={},
            file_sha256={},
            smoke={},
        )
        session.add(spec)
        session.flush()
        return spec.id


def payload():
    return {
        "spec_version": 1,
        "device": "a5",
        "language": "catlass",
        "source": {
            "url": "https://example.com/repo.git",
            "commit": "a" * 40,
            "path": "candidate",
            "dependencies": ["shared"],
        },
    }


def test_submit_and_status_survive_dispatch_outage(client, monkeypatch):
    seed(client)

    def unavailable(*args):
        raise RedisConnectionError("private credential")

    monkeypatch.setattr("cannery.submissions.api.publish_submission", unavailable)
    response = client.post("/api/v1/kernels/team/identity/submissions", json=payload())
    assert response.status_code == 202
    status = client.get(response.json()["status_url"])
    assert status.status_code == 200 and status.json()["status"] == "queued"
    assert status.json()["eta_seconds"] is None
    assert status.json()["stages"]["build"] == "pending"
    assert len(client.get("/api/v1/submissions/mine").json()) == 1
    assert "credential" not in status.text and "scratch_token" not in status.text


def test_revision_body_is_commit_only_and_old_request_remains_unchanged(client):
    seed(client)
    parent = client.post("/api/v1/kernels/team/identity/submissions", json=payload()).json()["id"]
    assert (
        client.post(
            f"/api/v1/submissions/{parent}/revisions", json={"commit": "b" * 40}
        ).status_code
        == 404
    )
    engine = client.app.state.test_engine
    with Session(engine) as session, session.begin():
        session.add(
            Candidate(
                id=UUID(parent),
                binding_id=UUID(parent),
                binding={},
                manifest={},
                snapshot={},
                content_sha256="b" * 64,
                file_sha256={},
            )
        )
        session.get(IntakeState, UUID(parent)).status = "awaiting_build"
    response = client.post(f"/api/v1/submissions/{parent}/revisions", json={"commit": "b" * 40})
    assert response.status_code == 202
    child = client.get(response.json()["status_url"]).json()
    before = client.get(f"/api/v1/submissions/{parent}").json()
    assert child["parent_id"] == parent and child["source"]["commit"] == "b" * 40
    assert before["source"]["commit"] == "a" * 40 and before["status"] == "awaiting_build"
    assert child["target"] == before["target"] and child["spec_version"] == before["spec_version"]
    assert (
        client.post(
            f"/api/v1/submissions/{parent}/revisions", json={"commit": "b" * 40, "device": "a3"}
        ).status_code
        == 422
    )
    assert (
        client.post(f"/api/v1/submissions/{parent}/revisions", json={"commit": "main"}).status_code
        == 422
    )


def test_namespace_and_non_git_boundaries(client):
    spec = seed(client)
    engine = client.app.state.test_engine
    assert (
        client.post("/api/v1/kernels/other/identity/submissions", json=payload()).status_code == 403
    )
    assert (
        client.post("/api/v1/kernels/team/missing/submissions", json=payload()).status_code == 404
    )
    for changed in (
        {"language": "triton"},
        {"device": "cpu"},
        {"source": {"type": "archive", "path": "/tmp/a.zip"}},
    ):
        assert (
            client.post(
                "/api/v1/kernels/team/identity/submissions", json=payload() | changed
            ).status_code
            == 422
        )
    key = client.post("/api/v1/kernels/team/identity/submissions", json=payload()).json()["id"]
    # Another namespace's request is not visible even with forged role headers.
    with Session(engine) as session, session.begin():
        other = Submission(
            namespace="other",
            spec_id=spec,
            source=payload()["source"],
            target={"device": "a5", "language": "catlass"},
        )
        session.add(other)
        session.flush()
        foreign = other.id
        session.add(IntakeState(id=foreign))
    assert (
        client.get(f"/api/v1/submissions/{foreign}", headers={"X-Role": "admin"}).status_code == 404
    )
    assert (
        client.post(
            f"/api/v1/submissions/{foreign}/revisions", json={"commit": "b" * 40}
        ).status_code
        == 404
    )
    assert [job["id"] for job in client.get("/api/v1/submissions/mine").json()] == [key]


def test_anonymous_submission_is_denied():
    from fastapi.testclient import TestClient

    from cannery.api import create_app
    from cannery.settings import Settings

    with TestClient(create_app(Settings(dev_identity_enabled=False))) as client:
        assert (
            client.post("/api/v1/kernels/team/identity/submissions", json=payload()).status_code
            == 401
        )


def test_database_failure_is_safe_503(client, monkeypatch):
    from sqlalchemy.exc import SQLAlchemyError

    seed(client)

    def unavailable(*args):
        raise SQLAlchemyError("private database credential")

    monkeypatch.setattr("cannery.submissions.api.new_submission", unavailable)
    response = client.post("/api/v1/kernels/team/identity/submissions", json=payload())
    assert response.status_code == 503 and "credential" not in response.text
