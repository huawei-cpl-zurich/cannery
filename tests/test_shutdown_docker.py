"""Verify explicit SIGTERM handling when Python really is container PID 1."""

import os
import subprocess
import time
from pathlib import Path
from uuid import uuid4

import pytest


@pytest.mark.skipif(
    os.environ.get("CANNERY_BACKUP_DOCKER") != "1", reason="opt-in Docker shutdown check"
)
def test_idle_pid_one_exits_without_forced_kill():
    name = "cannery-m7a-stop-" + uuid4().hex[:10]
    image = name + ":app"

    def docker(*args):
        return subprocess.check_output(["docker", *args], text=True, timeout=30).strip()

    try:
        subprocess.run(
            [
                "docker",
                "build",
                "--target",
                "app",
                "-t",
                image,
                str(Path(__file__).resolve().parents[1]),
            ],
            check=True,
            timeout=600,
        )
        program = (
            "import threading; from cannery.shutdown import stop_on_term; "
            "event=threading.Event(); restore=stop_on_term(event); "
            "print('ready',flush=True); event.wait(60); restore(); assert event.is_set()"
        )
        docker(
            "run",
            "-d",
            "--name",
            name,
            "--label",
            "cannery.task=m7a",
            "--read-only",
            "--network",
            "none",
            "--pids-limit",
            "64",
            "--memory",
            "128m",
            "--entrypoint",
            "python",
            image,
            "-c",
            program,
        )
        deadline = time.monotonic() + 20
        while "ready" not in docker("logs", name):
            if time.monotonic() > deadline:
                raise AssertionError("PID1 readiness deadline exceeded")
            time.sleep(0.1)
        docker("stop", "-t", "3", name)
        assert docker("inspect", "-f", "{{.State.ExitCode}}", name) == "0"
    finally:
        if docker("ps", "-aq", "--filter", f"name=^/{name}$"):
            docker("stop", "-t", "1", name)
            docker("rm", name)
        subprocess.run(["docker", "image", "rm", image], check=False, timeout=30)
