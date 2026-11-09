"""Bounded local subprocesses; no inherited credentials, proxies or Git config."""

import os
import signal
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryFile


class IntakeError(ValueError):
    pass


def clean_environment() -> dict[str, str]:
    return {
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "HOME": "/nonexistent",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_ALLOW_PROTOCOL": "https",
    }


def run(
    argv: list[str],
    *,
    cwd: Path | None = None,
    data: bytes | None = None,
    env: dict[str, str] | None = None,
    timeout: float = 30,
    max_output: int = 16 * 1024 * 1024,
) -> bytes:
    with TemporaryFile() as output, TemporaryFile() as errors:
        process = subprocess.Popen(
            [sys.executable, "-m", "cannery.git_intake.limits", *argv],
            cwd=cwd,
            env=clean_environment() if env is None else env,
            stdin=subprocess.PIPE,
            stdout=output,
            stderr=errors,
            start_new_session=True,
        )
        try:
            process.communicate(data, timeout=timeout)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate()
            raise IntakeError("Intake process interrupted or deadline exceeded") from None
        if process.returncode != 0 or output.tell() > max_output:
            raise IntakeError("Intake process failed or exceeded output budget")
        output.seek(0)
        return output.read()


def git(
    repository: Path, *args: str, data: bytes | None = None, env: dict[str, str] | None = None
) -> bytes:
    return run(
        [
            "git",
            "--literal-pathspecs",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "init.templateDir=",
            "-c",
            "protocol.allow=never",
            "-c",
            "protocol.https.allow=always",
            "-c",
            "gc.auto=0",
            "--git-dir",
            str(repository),
            *args,
        ],
        data=data,
        env=env,
    )
