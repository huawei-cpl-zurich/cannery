from datetime import UTC, datetime, timedelta

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from redis.exceptions import ConnectionError
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from cannery.accounts import auth, operator
from cannery.accounts.api import install_auth
from cannery.accounts.models import Credential
from cannery.database import Base
from cannery.settings import Settings


@pytest.fixture
def setup(monkeypatch):
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    operator.create(engine, "alice", "a long password!")
    operator.membership(engine, "alice", "team", grant=True)
    app = FastAPI()
    install_auth(
        app,
        Settings(auth_enabled=True, auth_cookie_secure=False, auth_origin="http://127.0.0.1"),
        engine,
    )
    monkeypatch.setattr("cannery.accounts.api.throttle", lambda *args: None)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        yield engine, client
    engine.dispose()


def login(client):
    reply = client.post(
        "/api/v1/auth/login",
        headers={"X-Cannery-Login": "1"},
        json={"username": "alice", "password": "a long password!"},
    )
    assert reply.status_code == 200, reply.text
    return {"X-CSRF-Token": reply.json()["csrf_token"]}


def test_login_csrf_and_expiry(setup):
    engine, client = setup
    assert client.get("/api/v1/auth/me").status_code == 401
    assert (
        client.post(
            "/api/v1/auth/login", json={"username": "alice", "password": "wrong"}
        ).status_code
        == 403
    )
    headers = login(client)
    assert client.get("/api/v1/auth/me").json()["namespace"] == "team"
    assert client.post("/api/v1/auth/logout").status_code == 403
    with Session(engine) as db, db.begin():
        row = db.scalar(select(Credential))
        assert len(row.digest) == 64 and row.digest != client.cookies["cannery_session"]
        row.expires_at = datetime.now(UTC) - timedelta(seconds=1)
    assert client.get("/api/v1/auth/me").status_code == 401
    assert headers["X-CSRF-Token"]


def test_namespace_bound_tokens_and_revocation(setup):
    engine, client = setup
    headers = login(client)
    assert (
        client.post("/api/v1/auth/tokens", headers=headers, json={"namespace": "other"}).status_code
        == 403
    )
    reply = client.post(
        "/api/v1/auth/tokens", headers=headers, json={"namespace": "team", "label": "test"}
    )
    assert reply.status_code == 201
    token = reply.json()["token"]
    bearer = {"Authorization": "Bearer " + token}
    assert client.get("/api/v1/auth/me", headers=bearer).status_code == 200
    assert (
        client.post(
            "/api/v1/auth/namespace", headers=bearer, json={"namespace": "other"}
        ).status_code
        == 403
    )
    assert (
        client.post("/api/v1/auth/tokens", headers=bearer, json={"namespace": "team"}).status_code
        == 403
    )
    tokens = client.get("/api/v1/auth/tokens", headers=headers).json()
    assert token not in str(tokens)
    assert (
        client.delete("/api/v1/auth/tokens/" + tokens[0]["id"], headers=headers).status_code == 204
    )
    assert client.get("/api/v1/auth/me", headers=bearer).status_code == 401
    operator.membership(engine, "alice", "team", grant=False)
    assert client.get("/api/v1/auth/me").status_code == 401


def test_password_logout_and_account_disable(setup):
    engine, client = setup
    headers = login(client)
    assert (
        client.post(
            "/api/v1/auth/password",
            headers=headers,
            json={"current_password": "wrong", "new_password": "new long password"},
        ).status_code
        == 401
    )
    assert client.post("/api/v1/auth/logout", headers=headers).status_code == 204
    assert client.get("/api/v1/auth/me").status_code == 401
    login(client)
    operator.configure(engine, "alice", enabled=False)
    assert client.get("/api/v1/auth/me").status_code == 401
    with pytest.raises(HTTPException) as error:
        auth.login(engine, "alice", "a long password!", "team")
    assert error.value.status_code == 401


def test_login_origin_and_errors(setup):
    engine, client = setup
    payload = {"username": "alice", "password": "wrong"}
    assert (
        client.post(
            "/api/v1/auth/login",
            headers={"X-Cannery-Login": "1", "Origin": "https://evil.invalid"},
            json=payload,
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/api/v1/auth/login", headers={"X-Cannery-Login": "1"}, json=payload
        ).status_code
        == 401
    )
    assert (
        client.get("/api/v1/auth/me", headers={"Authorization": "Basic secret"}).status_code == 401
    )


def test_throttle_is_bounded_and_fails_closed(monkeypatch):
    from starlette.requests import Request

    from cannery.accounts.api import throttle

    class Redis:
        count = 0
        broken = False

        @classmethod
        def from_url(cls, *args, **kwargs):
            return cls()

        def eval(self, *args):
            if self.broken:
                raise ConnectionError("private connection details")
            return self.count

        def close(self):
            pass

    monkeypatch.setattr("cannery.accounts.api.Redis", Redis)
    request = Request({"type": "http", "client": ("127.0.0.1", 123)})
    Redis.count = 11
    with pytest.raises(HTTPException) as error:
        throttle(request, "alice", Settings())
    assert error.value.status_code == 429
    Redis.broken = True
    with pytest.raises(HTTPException) as error:
        throttle(request, "alice", Settings())
    assert error.value.status_code == 503 and "private" not in error.value.detail


def test_invalid_password_reset_does_not_hash(setup, monkeypatch):
    engine, client = setup

    def forbidden_hash(value):
        raise AssertionError("Untrusted reset reached expensive hashing")

    monkeypatch.setattr("cannery.accounts.auth.password_hash", forbidden_hash)
    assert (
        client.post(
            "/api/v1/auth/password",
            headers={"Authorization": "Bearer forged"},
            json={"current_password": "old", "new_password": "new long password"},
        ).status_code
        == 401
    )


def test_malformed_csrf_returns_denied():
    from starlette.requests import Request

    from cannery.accounts.api import credential

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "headers": [(b"cookie", b"cannery_session=abc"), (b"x-csrf-token", b"\xff")],
        }
    )
    with pytest.raises(HTTPException) as error:
        credential(request)
    assert error.value.status_code == 403
