import json

import pytest
from release_fixtures import admission as admission
from release_fixtures import arguments as arguments
from release_fixtures import benchmark as benchmark
from release_fixtures import records as records
from release_fixtures import setup_executor as setup_executor
from test_release_catalog import releases as releases
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

    def forward(self, path, limit, body=None, *, params=None, deadline=None):
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

    def unavailable():
        raise ValueError("Simulated unavailable Ascend device")

    monkeypatch.setattr(cli, "detect_runtime", unavailable)
    result = CliRunner().invoke(app, ["get", "team/relu", "--cache", str(tmp_path)])
    assert result.exit_code == 1 and "Download unavailable" in result.output
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("configured", [False, True])
def test_hanging_detector_is_terminated(monkeypatch, tmp_path, configured):
    import os
    import time

    from cannery.releases.cli import detect_runtime

    stub = tmp_path / "cannery_kernels.py"
    stub.write_text("import time\ndef runtime_fingerprint(): time.sleep(60)\n")
    if not configured:
        monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    start = time.monotonic()
    with pytest.raises(ValueError, match="deadline exceeded"):
        detect_runtime(timeout=0.3)
    assert time.monotonic() - start < 3


@pytest.mark.parametrize("stalled", ["status", "final"])
def test_wait_timeout_bounds_stalled_requests(monkeypatch, stalled):
    import asyncio
    import time
    from uuid import uuid4

    import httpx

    from cannery.releases import cli

    calls = []

    async def response(request):
        calls.append(request)
        if request.method == "POST":
            return httpx.Response(202, json={"id": str(uuid4())})
        if stalled == "final" and "/publication-jobs/" in request.url.path:
            await asyncio.sleep(0.5)
            return httpx.Response(200, json={"status": "complete"})
        await asyncio.sleep(60)
        return httpx.Response(200, json={"status": "complete"})

    client = ReleaseClient("http://testserver", transport=httpx.MockTransport(response))
    monkeypatch.setattr(cli, "ReleaseClient", lambda _: client)
    start = time.monotonic()
    result = CliRunner().invoke(app, ["publish", str(uuid4()), "--wait", "--timeout", "1"])
    assert result.exit_code == 1 and time.monotonic() - start < 3
    assert len(calls) == (2 if stalled == "status" else 3)


def test_ambiguous_cohort_choices_are_visible_in_info_and_get(releases, monkeypatch, tmp_path):
    import httpx

    from cannery.releases import cli
    from cannery.releases.download_api import install_downloads
    from cannery.settings import Settings

    engine, add, api, manifest = releases
    first, second = add(0.5), add(0.6, driver="other")
    install_downloads(api.app, Settings(release_enabled=True, dev_identity_enabled=True), engine)

    def response(request):
        result = api.request(
            request.method,
            str(request.url),
            json=json.loads(request.content) if request.content else None,
        )
        return httpx.Response(result.status_code, content=result.content)

    client = ReleaseClient("http://testserver", transport=httpx.MockTransport(response))
    monkeypatch.setattr(cli, "ReleaseClient", lambda _: client)
    runtime = tmp_path / "runtime.json"
    runtime.write_text(json.dumps(manifest.cohort.runtime))
    commands = [
        ["info", manifest.kernel],
        [
            "get",
            manifest.kernel,
            "--no-detect",
            "--runtime-json",
            str(runtime),
            "--cache",
            str(tmp_path / "cache"),
        ],
    ]
    for command in commands:
        result = CliRunner().invoke(app, command)
        assert result.exit_code == 1 and "cohort_required" in result.output
        assert f"--cohort {first[2]}" in result.output and f"--cohort {second[2]}" in result.output


@pytest.mark.parametrize("runtime", [[], None, 7, "fingerprint", {"soc": None}, {"soc": 123}])
def test_malformed_runtime_json_has_bounded_useful_cli_error(tmp_path, runtime):
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(runtime))
    result = CliRunner().invoke(
        app,
        [
            "get",
            "team/relu",
            "--no-detect",
            "--runtime-json",
            str(path),
            "--cache",
            str(tmp_path / "cache"),
        ],
    )
    assert result.exit_code == 1 and "Download unavailable or incompatible" in result.output
    assert not isinstance(result.exception, AttributeError)
    assert not (tmp_path / "cache").exists()


def test_automatic_a3_selection_uses_validated_runtime(tmp_path, monkeypatch):
    runtime = dict(
        machine="aarch64",
        python="3.11",
        torch="2.10.0",
        torch_npu="2.10.0",
        cann="9.1.0",
        soc="Ascend910B4-1",
    )
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(runtime))
    selections = []

    def unavailable(self, selection, cache):
        selections.append(selection)
        raise ValueError("offline")

    monkeypatch.setattr(ReleaseClient, "get", unavailable)
    result = CliRunner().invoke(
        app, ["get", "team/relu", "--no-detect", "--runtime-json", str(path)]
    )
    assert result.exit_code == 1 and len(selections) == 1
    assert selections[0].device == "a3" and selections[0].runtime == runtime
