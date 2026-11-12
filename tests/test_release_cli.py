import json

import pytest
from release_fixtures import admission as admission
from release_fixtures import arguments as arguments
from release_fixtures import benchmark as benchmark
from release_fixtures import records as records
from release_fixtures import setup_executor as setup_executor
from test_release_resolution import published as published
from typer.testing import CliRunner

from cannery.cli import app
from cannery.releases.client import ReleaseClient


@pytest.fixture
def cli_server(published, monkeypatch):
    _, _, _, m, identifier, server = published
    from cannery.releases.api import install_publications
    from cannery.releases.discovery_api import install_discovery
    from cannery.settings import Settings

    engine, key, _, _, _, _ = published
    settings = Settings(dev_identity_enabled=True, dev_namespace="team", release_enabled=True)
    install_discovery(server.app, settings, engine)
    install_publications(server.app, settings, engine)

    def forward(self, path, limit, body=None, *, params=None):
        response = server.request("GET" if body is None else "POST", path, json=body, params=params)
        response.raise_for_status()
        assert len(response.content) <= limit
        return response.content

    monkeypatch.setattr(ReleaseClient, "bytes", forward)
    return m, key, identifier


def test_get_info_source_and_withdraw_cli(cli_server, tmp_path):
    m, key, identifier = cli_server
    runner = CliRunner()
    runtime = tmp_path / "runtime.json"
    runtime.write_text(json.dumps(m.cohort.runtime))
    result = runner.invoke(
        app,
        [
            "get",
            m.kernel,
            "--no-detect",
            "--runtime-json",
            str(runtime),
            "--cache",
            str(tmp_path / "cache"),
            "--shape",
            "2,3",
        ],
    )
    assert result.exit_code == 0, result.output
    saved = json.loads(result.output)
    assert saved["package_sha256"] == m.package.sha256
    assert runner.invoke(app, ["info", m.kernel]).exit_code == 0
    source = runner.invoke(app, ["source", str(identifier)])
    assert source.exit_code == 0 and json.loads(source.output)["source"] == m.source.model_dump(
        mode="json"
    )
    status = runner.invoke(app, ["publication-status", str(key)])
    assert status.exit_code == 0 and "published" in status.output
    result = runner.invoke(app, ["withdraw", str(key)])
    assert result.exit_code == 0 and "withdrawn" in result.output
    assert runner.invoke(app, ["source", str(identifier)]).exit_code == 1
    assert runner.invoke(app, ["publish", str(key), "--wait", "--timeout", "1"]).exit_code == 1
    # Worker is intentionally not running; bounded waiting is tested below with terminal jobs.


def test_cpu_download_requires_explicit_runtime_file(tmp_path):
    runner = CliRunner()
    result = runner.invoke(app, ["get", "team/relu", "--no-detect", "--cache", str(tmp_path)])
    assert result.exit_code == 1
    assert not list(tmp_path.rglob("package.tar"))


def test_publish_observes_terminal_failure_without_reposting(monkeypatch):
    from uuid import uuid4

    calls = []
    identifier, job = uuid4(), uuid4()

    def json_reply(self, path, body=None, **kwargs):
        calls.append((path, body))
        return (
            {"id": str(job)}
            if body is not None
            else {"status": "failed", "error": "release_artifact_unavailable"}
        )

    monkeypatch.setattr(ReleaseClient, "json", json_reply)
    result = CliRunner().invoke(app, ["publish", str(identifier), "--wait"])
    assert result.exit_code == 1
    assert sum(body is not None for _, body in calls) == 1


def test_device_detection_failure_is_bounded_and_does_not_download(monkeypatch, tmp_path):
    from cannery.releases import cli

    def unavailable(_):
        raise RuntimeError("Simulated unavailable Ascend device")

    monkeypatch.setattr(cli.importlib, "import_module", unavailable)
    result = CliRunner().invoke(app, ["get", "team/relu", "--cache", str(tmp_path)])
    assert result.exit_code == 1 and "Download unavailable" in result.output
    assert not list(tmp_path.iterdir())
