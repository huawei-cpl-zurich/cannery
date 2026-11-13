from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from cannery.accounts import auth, operator
from cannery.accounts.context import CURRENT
from cannery.api import create_app
from cannery.database import Base
from cannery.git_intake import GitSource
from cannery.registry.registrations import create_registration
from cannery.settings import Settings


@pytest.fixture
def setup(monkeypatch):
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    for name, namespace in [("alice", "team"), ("bob", "other"), ("carol", "team")]:
        operator.create(engine, name, "a long password!")
        operator.membership(engine, name, namespace, grant=True)
    operator.create(engine, "operator", "a long password!", "admin")
    operator.membership(engine, "operator", "team", grant=True)
    monkeypatch.setattr("cannery.api.make_engine", lambda settings: engine)
    app = create_app(Settings(auth_enabled=True, auth_cookie_secure=False))
    source = GitSource(url="https://github.com/example/kernel", commit="a" * 40, path="kernel")
    identifier = create_registration(engine, source, "team", "toy", False)
    with TestClient(app) as client:
        yield engine, client, identifier


def signed(client, engine, username):
    raw = auth.login(engine, username, "a long password!", None)
    client.cookies.set("cannery_session", raw)
    return {"X-CSRF-Token": auth.csrf(raw)}


def test_shared_namespace_and_context_reset(setup):
    engine, client, identifier = setup
    for user in ["alice", "carol"]:
        signed(client, engine, user)
        assert client.get(f"/api/v1/registrations/{identifier}").status_code == 200
        assert len(client.get("/api/v1/registrations").json()) == 1
        assert client.get("/register-kernel").status_code == 200
        assert client.get("/admin").status_code == 403
        assert CURRENT.get() is None
    signed(client, engine, "bob")
    assert client.get(f"/api/v1/registrations/{identifier}").status_code == 404
    assert client.get("/api/v1/registrations").json() == []
    assert client.get("/admin").status_code == 403
    signed(client, engine, "operator")
    assert client.get("/admin").status_code == 200
    client.cookies.clear()
    assert client.get("/api/v1/registrations").status_code == 401
    assert client.get("/global-best").status_code == 200


@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/kernels",
        "/api/v1/kernels/team/toy/submissions",
        "/api/v1/submissions/{id}/revise",
        "/api/v1/submissions/{id}/publish",
        "/api/v1/submissions/{id}/withdraw",
        "/api/v1/artifacts/resolve",
        "/register-kernel",
        "/submit-implementation",
        "/submissions/{id}/revise",
        "/submissions/{id}/publish",
    ],
)
def test_all_mutations_require_authentication_and_cookie_csrf(setup, path):
    engine, client, _ = setup
    path = path.replace("{id}", str(uuid4()))
    assert client.post(path, json={}).status_code == 401
    signed(client, engine, "alice")
    assert client.post(path, json={}).status_code == 403


def test_permission_revocation_takes_effect_on_next_request(setup):
    engine, client, identifier = setup
    signed(client, engine, "alice")
    assert client.get(f"/api/v1/registrations/{identifier}").status_code == 200
    operator.membership(engine, "alice", "team", grant=False)
    assert client.get(f"/api/v1/registrations/{identifier}").status_code == 401
    # Expired/revoked sessions do not prevent anonymous published browsing.
    assert client.get("/global-best").status_code == 200


def test_development_and_account_modes_are_exclusive():
    with pytest.raises(ValueError, match="cannot be combined"):
        create_app(Settings(auth_enabled=True, dev_identity_enabled=True))
