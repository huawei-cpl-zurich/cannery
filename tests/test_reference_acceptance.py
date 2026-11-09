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
