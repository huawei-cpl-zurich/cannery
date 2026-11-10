import pytest
from fastapi.testclient import TestClient

from cannery.api import create_app
from cannery.settings import Settings


@pytest.mark.parametrize(
    "enabled,role,expected",
    [
        (False, "user", 401),
        (False, "admin", 401),
        (True, "user", 403),
        (True, "admin", 200),
    ],
)
def test_admin_roles(enabled, role, expected) -> None:
    with TestClient(create_app(Settings(dev_identity_enabled=enabled, dev_role=role))) as client:
        assert client.get("/admin", headers={"X-Role": "admin"}).status_code == expected
        assert ("Administration</a>" in client.get("/global-best").text) == (expected == 200)


def test_navigation_and_local_assets() -> None:
    with TestClient(create_app(Settings(dev_identity_enabled=True))) as client:
        for page in ("global-best", "my-submissions", "register-kernel", "submit-implementation"):
            response = client.get(f"/{page}")
            assert response.status_code == 200
            assert "Development identity enabled" in response.text
            assert "hx-get=" in response.text
        assert client.get("/static/htmx.min.js").status_code == 200
        assert client.get("/static/style.css").status_code == 200
        assert client.get("/unknown").status_code == 404
    with TestClient(create_app()) as client:
        assert client.get("/my-submissions").status_code == 401
        assert client.post("/register-kernel", json={}).status_code == 401
