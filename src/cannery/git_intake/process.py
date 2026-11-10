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
    env = {
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "HOME": "/nonexistent",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_ALLOW_PROTOCOL": "https",
    }
    if os.environ.get("CANNERY_ATTEMPT_GROUP") == str(os.getpgrp()):
        env["CANNERY_ATTEMPT_GROUP"] = str(os.getpgrp())
    return env


def attempt_budget(seconds: int = 140) -> None:
    """Called only inside a fresh supervised attempt session, never in the API."""
    if not 1 <= seconds <= 140:
        raise IntakeError("Invalid attempt deadline")
    if os.getpid() != os.getpgrp():
        raise IntakeError("Attempt must run in its own process session")
    os.environ["CANNERY_ATTEMPT_GROUP"] = str(os.getpgrp())
    signal.signal(signal.SIGALRM, lambda *_: os.killpg(os.getpgrp(), signal.SIGKILL))
    signal.alarm(seconds)


def run(
    argv: list[str],
    *,
    cwd: Path | None = None,
    data: bytes | None = None,
    env: dict[str, str] | None = None,
    timeout: float = 30,
    max_output: int = 16 * 1024 * 1024,
    git_limits: bool = True,
) -> bytes:
    nested = os.environ.get("CANNERY_ATTEMPT_GROUP") == str(os.getpgrp())
    with TemporaryFile() as output, TemporaryFile() as errors:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "cannery.git_intake.limits",
                *([] if git_limits else ["--coordinator"]),
                *argv,
            ],
            cwd=cwd,
            env=clean_environment() if env is None else env,
            stdin=subprocess.PIPE,
            stdout=output,
            stderr=errors,
            start_new_session=not nested,
        )
        try:
            process.communicate(data, timeout=timeout)
        except BaseException:
            try:
                os.killpg(os.getpgrp() if nested else process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate()
            raise IntakeError("Intake process interrupted or deadline exceeded") from None
        if process.returncode != 0 or output.tell() > max_output:
            if nested:
                os.killpg(os.getpgrp(), signal.SIGKILL)
            raise IntakeError("Intake process failed or exceeded output budget")
        output.seek(0)
        return output.read()


def git(
    repository: Path,
    *args: str,
    data: bytes | None = None,
    env: dict[str, str] | None = None,
    max_output: int = 16 * 1024 * 1024,
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
        max_output=max_output,
    )
