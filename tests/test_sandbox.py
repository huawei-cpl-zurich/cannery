import subprocess
from unittest.mock import MagicMock

import pytest
from spec_fixtures import spec_files

from cannery.registry.manifest import validate_spec
from cannery.registry.sandbox import SmokeError, run_smoke


def test_missing_image_never_runs_source(monkeypatch):
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "docker")

    monkeypatch.setattr("cannery.registry.sandbox.subprocess.run", fail)
    monkeypatch.setattr(
        "cannery.registry.sandbox.subprocess.Popen",
        lambda *a, **kw: pytest.fail("must not execute"),
    )
    files = spec_files(reference=b"raise RuntimeError('not in API')")
    with pytest.raises(SmokeError, match="unavailable"):
        run_smoke(validate_spec(files, "spec"), files, "spec", image="missing")


def test_deadline_removes_restricted_container(monkeypatch):
    inspected = MagicMock(stdout="sha256:" + "a" * 64)
    monkeypatch.setattr("cannery.registry.sandbox.subprocess.run", lambda *a, **kw: inspected)
    process = MagicMock()
    process.poll.return_value = None
    calls = []

    def popen(argv, **kwargs):
        calls.append(argv)
        return process

    monkeypatch.setattr("cannery.registry.sandbox.subprocess.Popen", popen)
    removed = []
    monkeypatch.setattr("cannery.registry.sandbox.remove_smoke", removed.append)
    files = spec_files()
    with pytest.raises(SmokeError):
        run_smoke(validate_spec(files, "spec"), files, "spec", image="trusted", timeout=0.001)
    assert removed and removed[0].startswith("cannery-smoke-")
    assert "none" in calls[0] and "--read-only" in calls[0]
    assert "--cap-drop=ALL" in calls[0] and "--pids-limit" in calls[0]
    assert not any("socket" in item or "CANNERY_" in item for item in calls[0][:-1])
    process.kill.assert_called_once()
    process.wait.assert_called_once()


@pytest.mark.parametrize("timeout", [0, float("inf"), float("nan")])
def test_invalid_deadlines_do_not_start_containers(timeout):
    files = spec_files()
    with pytest.raises(SmokeError, match="deadline"):
        run_smoke(validate_spec(files, "spec"), files, "spec", image="trusted", timeout=timeout)


def test_mismatched_source_is_rejected_before_docker(monkeypatch):
    files = spec_files()
    spec = validate_spec(files, "spec")
    files["spec/reference.py"] = b"changed"
    monkeypatch.setattr(
        "cannery.registry.sandbox.subprocess.run",
        lambda *a, **k: pytest.fail("must reject before Docker"),
    )
    with pytest.raises(SmokeError, match="does not match"):
        run_smoke(spec, files, "spec", image="trusted")


def test_failed_removal_exposes_owned_handle_for_recovery(monkeypatch):
    from cannery.registry.sandbox import remove_smoke

    failed = MagicMock(returncode=1, stdout="")
    monkeypatch.setattr("cannery.registry.sandbox.subprocess.run", lambda *a, **k: failed)
    handle = "cannery-smoke-" + "a" * 32
    with pytest.raises(SmokeError) as caught:
        remove_smoke(handle)
    assert caught.value.handle == handle
    assert handle in str(caught.value)


def test_smoke_observations_never_claim_independent_execution_proof():
    from cannery.registry.freeze import SmokeEvidence

    evidence = SmokeEvidence(
        spec_sha256="a" * 64,
        image_id="sha256:" + "b" * 64,
        python="3.12",
        torch="2.6.0+cpu",
        cases=1,
    )
    assert evidence.model_dump()["assurance"] == "diagnostic"
