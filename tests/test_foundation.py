from fastapi.testclient import TestClient
from typer.testing import CliRunner

from cannery.api import create_app
from cannery.cli import app
from cannery.settings import Settings


def test_live_without_dependencies() -> None:
    with TestClient(create_app()) as client:
        assert client.get("/health/live").json() == {"status": "ok"}


def test_settings_environment(monkeypatch) -> None:
    monkeypatch.setenv("CANNERY_SERVICE_NAME", "test-registry")
    assert Settings().service_name == "test-registry"


def test_cli_help_and_version() -> None:
    runner = CliRunner()
    assert runner.invoke(app, ["--help"]).exit_code == 0
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "0.1.0"
