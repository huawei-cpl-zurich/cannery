import json

import pytest
from typer.testing import CliRunner

from cannery.accounts import client
from cannery.cli import app


@pytest.fixture(autouse=True)
def private_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))


def test_private_origin_bound_credentials():
    first, other = "http://127.0.0.1:8095", "http://127.0.0.1:8096"
    assert client.headers(first) == {}
    client.save(first, "a" * 43)
    assert client.location(first).stat().st_mode & 0o777 == 0o600
    assert client.headers(first) == {"Authorization": "Bearer " + "a" * 43}
    assert client.headers(other) == {}
    client.location(first).chmod(0o644)
    with pytest.raises(ValueError, match="private"):
        client.headers(first)


@pytest.mark.parametrize(
    "url",
    [
        "http://remote.invalid",
        "https://user:secret@example.com",
        "https://example.com?token=secret",
        "https://example.com/#secret",
        "ftp://example.com",
    ],
)
def test_insecure_or_credential_containing_origins_are_rejected(url):
    with pytest.raises(ValueError):
        client.save(url, "a" * 43)


def test_tampered_or_symlinked_credentials_fail_closed(tmp_path):
    url = "https://example.com"
    client.save(url, "a" * 43)
    path = client.location(url)
    path.write_text(json.dumps({"origin": "https://other.invalid", "token": "b" * 43}))
    with pytest.raises(ValueError):
        client.headers(url)
    path.unlink()
    target = tmp_path / "target"
    target.write_text("private")
    path.symlink_to(target)
    with pytest.raises(ValueError):
        client.headers(url)


def test_cli_login_never_accepts_remote_cleartext_passwords(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Credentials reached an unsafe connection")

    monkeypatch.setattr("cannery.accounts.client_cli.httpx.Client", forbidden)
    response = CliRunner().invoke(
        app, ["login", "--url", "http://remote.invalid", "--username", "alice"]
    )
    assert response.exit_code == 1 and "Sign-in failed" in response.output


def test_logout_preserves_credentials_on_network_failure(monkeypatch):
    import httpx

    url = "https://example.com"
    client.save(url, "a" * 43)

    def unavailable(*args, **kwargs):
        raise httpx.ConnectError("sensitive connection details")

    monkeypatch.setattr("cannery.accounts.client_cli.httpx.post", unavailable)
    response = CliRunner().invoke(app, ["logout", "--url", url])
    assert response.exit_code == 1 and client.location(url).exists()
    assert "sensitive" not in response.output


def test_private_directory_is_required():
    url = "https://example.com"
    client.save(url, "a" * 43)
    client.location(url).parent.chmod(0o777)
    with pytest.raises(ValueError, match="directory"):
        client.headers(url)
    with pytest.raises(ValueError, match="directory"):
        client.save(url, "b" * 43)
