import runpy
import subprocess
from pathlib import Path

import pytest


def test_fixture_creation_timeout_always_reconciles_ownership(monkeypatch, capsys):
    script = Path(__file__).resolve().parents[1] / "scripts/reference_acceptance.py"
    check = runpy.run_path(str(script), run_name="reference_acceptance")["check_collision"]
    removed = []
    monkeypatch.setitem(check.__globals__, "remove_smoke", removed.append)

    def failed_create(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 5)

    monkeypatch.setattr(subprocess, "run", failed_create)
    with pytest.raises(subprocess.TimeoutExpired):
        check(None, {}, "fixture-image")
    assert len(removed) == 1 and removed[0].startswith("cannery-smoke-")
    assert "@" in removed[0]
    assert removed[0] in capsys.readouterr().out


def test_failed_image_removal_fails_acceptance(monkeypatch):
    from unittest.mock import MagicMock

    script = Path(__file__).resolve().parents[1] / "scripts/reference_acceptance.py"
    remove = runpy.run_path(str(script), run_name="reference_acceptance")["remove_image"]

    def unavailable(argv, **kwargs):
        return MagicMock(returncode=0, stdout="still-present" if argv[2] == "ls" else "")

    monkeypatch.setattr(subprocess, "run", unavailable)
    with pytest.raises(RuntimeError, match="fixture-image"):
        remove("fixture-image")


def test_cleanup_failure_is_never_expected_rejection(monkeypatch):
    from cannery.registry.sandbox import CleanupError

    script = Path(__file__).resolve().parents[1] / "scripts/reference_acceptance.py"
    reject = runpy.run_path(str(script), run_name="reference_acceptance")["expect_rejection"]
    handle = "cannery-smoke-" + "a" * 32 + "@" + "b" * 32
    monkeypatch.setitem(reject.__globals__, "new_handle", lambda: handle)
    monkeypatch.setitem(reject.__globals__, "validate_spec", lambda *a: None)

    def failed(*args, **kwargs):
        raise CleanupError("injected cleanup failure", handle=handle)

    monkeypatch.setitem(reject.__globals__, "run_smoke", failed)
    removed = []
    monkeypatch.setitem(reject.__globals__, "remove_smoke", removed.append)
    with pytest.raises(CleanupError):
        reject({}, "fixture-image")
    assert removed == [handle]
