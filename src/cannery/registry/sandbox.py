"""Trusted coordinator: author code runs only in a separate restricted container."""

import base64
import json
import math
import re
import subprocess
import time
from pathlib import Path
from tempfile import TemporaryFile
from uuid import uuid4

from cannery.git_intake.snapshots import MAX_BYTES, MAX_FILES
from cannery.registry.freeze import SmokeEvidence
from cannery.registry.manifest import SpecError, ValidatedSpec, validate_spec


class SmokeError(ValueError):
    def __init__(self, message: str, handle: str | None = None) -> None:
        self.handle = handle
        super().__init__(message if handle is None else f"{message}; sandbox={handle}")


def remove_smoke(name: str) -> None:
    if not re.fullmatch(r"cannery-smoke-[a-f0-9]{32}", name):
        raise SmokeError("Invalid sandbox handle")
    for _ in range(2):
        try:
            subprocess.run(
                ["docker", "rm", "-f", name], capture_output=True, timeout=3, check=False
            )
            remaining = subprocess.run(
                [
                    "docker",
                    "ps",
                    "-a",
                    "--filter",
                    f"name=^/{name}$",
                    "--format",
                    "{{.ID}}",
                ],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
            if remaining.returncode == 0 and not remaining.stdout.strip():
                return
        except (OSError, subprocess.SubprocessError):
            pass
    raise SmokeError("Sandbox cleanup incomplete; retry removal using this handle", handle=name)


def run_smoke(
    spec: ValidatedSpec,
    files: dict[str, bytes],
    path: str,
    *,
    image: str,
    name: str | None = None,
    timeout: float = 30,
) -> SmokeEvidence:
    if not math.isfinite(timeout) or timeout <= 0 or timeout > 60:
        raise SmokeError("Smoke deadline must be finite and between 0 and 60 seconds")
    if len(files) > MAX_FILES or sum(map(len, files.values())) > MAX_BYTES:
        raise SmokeError("Smoke source exceeds admission budget")
    try:
        if validate_spec(files, path) != spec:
            raise SpecError("Source mismatch")
    except SpecError:
        raise SmokeError("Smoke source does not match validated specification") from None
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9./:@_-]{0,255}", image):
        raise SmokeError("Invalid configured image")
    name = name if name is not None else "cannery-smoke-" + uuid4().hex
    if not re.fullmatch(r"cannery-smoke-[a-f0-9]{32}", name):
        raise SmokeError("Invalid sandbox handle")
    try:
        inspected = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", image],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        image_id = inspected.stdout.strip()
        if not re.fullmatch(r"sha256:[a-f0-9]{64}", image_id):
            raise SmokeError("Smoke image did not resolve to an immutable image ID")
    except (OSError, subprocess.SubprocessError):
        raise SmokeError("Configured smoke image is unavailable") from None
    payload = json.dumps(
        {
            "files": {name: base64.b64encode(data).decode() for name, data in files.items()},
            "path": path,
            "spec": spec.manifest.spec.model_dump(mode="json"),
            "environment": spec.environment.model_dump(mode="json"),
            "timeout": timeout,
        }
    ).encode()
    runner = Path(__file__).with_name("reference_runner.py").read_text()
    with TemporaryFile() as stdin, TemporaryFile() as output:
        stdin.write(payload)
        stdin.seek(0)
        process = subprocess.Popen(
            [
                "docker",
                "run",
                "--rm",
                "--name",
                name,
                "--network",
                "none",
                "--read-only",
                "--cap-drop=ALL",
                "--cap-add=SETUID",
                "--cap-add=SETGID",
                "--cap-add=KILL",
                "--security-opt",
                "no-new-privileges",
                "--memory",
                "512m",
                "--memory-swap",
                "512m",
                "--cpus",
                "1",
                "--pids-limit",
                "64",
                "--tmpfs",
                "/work:rw,nosuid,nodev,size=32m,uid=0,gid=0,mode=755",
                "--tmpfs",
                "/tmp:rw,nosuid,nodev,size=32m",
                "--user",
                "0:0",
                "--log-driver",
                "none",
                "-i",
                image_id,
                "python",
                "-I",
                "-c",
                runner,
            ],
            stdin=stdin,
            stdout=output,
            stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + timeout
        try:
            while process.poll() is None:
                if time.monotonic() >= deadline or output.tell() > 1024 * 1024:
                    raise SmokeError("Reference smoke exceeded time or output budget")
                time.sleep(0.05)
            if process.returncode != 0 or output.tell() > 1024 * 1024:
                raise SmokeError("Reference smoke failed")
            output.seek(0)
            result = json.loads(output.read())
            return SmokeEvidence(**result, image_id=image_id, spec_sha256=spec.content_sha256)
        except (ValueError, TypeError):
            raise SmokeError("Reference smoke failed or returned invalid evidence") from None
        finally:
            try:
                remove_smoke(name)
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=5)
