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
