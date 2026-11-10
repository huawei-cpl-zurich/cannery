from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from redis.exceptions import ConnectionError as RedisConnectionError
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from cannery.api import create_app
from cannery.database import Base
from cannery.settings import Settings


@pytest.fixture
def client(monkeypatch):
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    monkeypatch.setattr("cannery.api.make_engine", lambda _: engine)
    monkeypatch.setattr("cannery.registry.api.publish_registration", MagicMock())
    with TestClient(create_app(Settings(dev_identity_enabled=True))) as client:
        client.app.state.test_engine = engine
        yield client
    engine.dispose()


def payload():
    return {"source": {"url": "https://example.com/repo.git", "commit": "a" * 40, "path": "spec/"}}


def test_registration_dispatch_outage_is_durable(client, monkeypatch):
    def unavailable(*args):
        raise RedisConnectionError("secret")

    monkeypatch.setattr("cannery.registry.api.publish_registration", unavailable)
    response = client.post("/api/v1/kernels", json=payload())
    assert response.status_code == 202
    assert client.get(response.json()["status_url"]).json()["status"] == "queued"
    assert len(client.get("/api/v1/registrations").json()) == 1


def test_namespace_role_and_source_upload_boundaries(client):
    body = payload() | {"namespace": "other"}
    assert client.post("/api/v1/kernels", json=body).status_code == 403
    body = payload() | {"allow_new_version": True}
    assert client.post("/api/v1/kernels", json=body, headers={"X-Role": "admin"}).status_code == 422
    body = payload()
    body["source"] = {"type": "archive", "path": "/tmp/source.zip"}
    assert client.post("/api/v1/kernels", json=body).status_code == 422
    assert client.post("/api/v1/uploads", json={}).status_code == 404
    assert client.get("/api/v1/kernels/other/name/specs/1").status_code == 404


def test_anonymous_requests_are_denied():
    with TestClient(create_app()) as client:
        assert client.post("/api/v1/kernels", json=payload()).status_code == 401


def test_job_scope_and_role_are_server_owned(client):
    from sqlalchemy.orm import Session

    from cannery.git_intake import GitSource
    from cannery.registry.registrations import Registration, create_registration

    engine = client.app.state.test_engine
    response = client.post("/api/v1/kernels", json=payload(), headers={"X-Role": "admin"})
    from uuid import UUID

    with Session(engine) as session:
        row = session.get(Registration, UUID(response.json()["id"]))
        assert row.namespace == "team" and row.allow_new_version is False
    foreign = create_registration(
        engine, GitSource.model_validate(payload()["source"]), "other", None, False
    )
    assert client.get(f"/api/v1/registrations/{foreign}").status_code == 404
    assert all(job["id"] != str(foreign) for job in client.get("/api/v1/registrations").json())


def test_database_outage_is_safe_and_retryable(client, monkeypatch):
    from sqlalchemy.exc import SQLAlchemyError

    def unavailable(*args):
        raise SQLAlchemyError("private connection credential")

    monkeypatch.setattr("cannery.registry.api.create_registration", unavailable)
    response = client.post("/api/v1/kernels", json=payload())
    assert response.status_code == 503 and "credential" not in response.text


def test_web_registration_queues_and_polls_owned_status(client):
    form = {
        "url": "https://example.com/repo.git",
        "commit": "a" * 40,
        "path": "spec/",
        "dependencies": "shared",
        "name": "identity",
    }
    response = client.post(
        "/register-kernel",
        data=form,
        headers={"Origin": "http://testserver"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    status = client.get(response.headers["location"])
    assert status.status_code == 200 and 'hx-trigger="every 2s"' in status.text
    assert "queued" in status.text
    assert "cleanup_handle" not in status.text and "internal_snapshot" not in status.text


def test_web_form_validation_and_same_origin(client):
    assert '<form method="post"' in client.get("/register-kernel").text
    assert client.post("/register-kernel", data={"path": "spec"}).status_code == 403
    assert (
        client.post(
            "/register-kernel", data={}, headers={"Origin": "https://evil.example"}
        ).status_code
        == 403
    )
    response = client.post(
        "/register-kernel",
        data={"url": "<script>alert(1)</script>"},
        headers={"Origin": "http://testserver"},
    )
    assert response.status_code == 200 and "Enter a Git repository" in response.text
    assert "<script>alert" not in response.text
    response = client.post(
        "/register-kernel",
        content="x=" + "a" * 17000,
        headers={
            "Origin": "http://testserver",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    assert response.status_code == 413


def test_web_status_stops_polling_at_terminal_state_and_hides_foreign_job(client):
    from uuid import UUID

    from sqlalchemy.orm import Session

    from cannery.registry.registrations import Registration

    key = client.post("/api/v1/kernels", json=payload()).json()["id"]
    with Session(client.app.state.test_engine) as session, session.begin():
        job = session.get(Registration, UUID(key))
        job.status, job.error = "failed", "reference_failed"
    response = client.get(f"/registrations/{key}")
    assert "reference_failed" in response.text and 'hx-trigger="every 2s"' not in response.text
    with Session(client.app.state.test_engine) as session, session.begin():
        session.get(Registration, UUID(key)).namespace = "other"
    assert client.get(f"/registrations/{key}").status_code == 404
