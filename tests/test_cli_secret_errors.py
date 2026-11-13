"""Unexpected dependency failures must not print password-bearing CLI locals."""

import os
import subprocess
import sys
from pathlib import Path


def test_unexpected_login_failure_never_prints_password_locals(tmp_path):
    password = "private-fixture-login-password"
    path = tmp_path / "password"
    path.write_text(password)
    path.chmod(0o600)
    script = """from cannery.cli import app
import cannery.accounts.client_cli as cli

def fail(*args, **kwargs):
    raise RuntimeError('injected unexpected dependency failure')

cli.httpx.Client = fail
app()
"""
    root = Path(__file__).resolve().parents[1]
    response = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            "login",
            "--url",
            "https://example.com",
            "--username",
            "alice",
            "--password-file",
            str(path),
        ],
        cwd=root,
        env=os.environ | {"PYTHONPATH": str(root / "src")},
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert response.returncode != 0
    assert "injected unexpected dependency failure" in response.stderr
    assert password not in response.stdout + response.stderr
