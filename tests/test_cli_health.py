import httpx
import pytest
from typer.testing import CliRunner

from cannery.cli import app


@pytest.mark.parametrize(
    "status,body,expected",
    [
        (200, {"status": "ok"}, 0),
        (503, {"status": "unavailable"}, 1),
        (200, {"status": "wrong"}, 1),
        (200, [], 1),
    ],
)
def test_health_exit_codes(monkeypatch, status, body, expected) -> None:
    def get(url, **kwargs):
        assert kwargs["timeout"] == 5
        return httpx.Response(status, json=body, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", get)
    assert CliRunner().invoke(app, ["health", "--url", "http://test"]).exit_code == expected


def test_health_timeout_is_redacted(monkeypatch) -> None:
    def get(*args, **kwargs):
        raise httpx.ReadTimeout("secret")

    monkeypatch.setattr(httpx, "get", get)
    result = CliRunner().invoke(app, ["health"])
    assert result.exit_code == 1
    assert "secret" not in result.output
