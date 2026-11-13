"""Retained CPL jobs and resumable registered SSH/remote-agent transfers."""

import base64
import gzip
import hashlib
import json
import os
import re
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from cannery.git_intake.process import IntakeError, run


class RemotePending(Exception):
    """Transport lost or dispatch ambiguous: retain the attempt and observe it."""


class RemoteFailure(Exception):
    def __init__(self, exit_code: int | None) -> None:
        self.exit_code = exit_code
        super().__init__("Retained remote operation failed")


def durable_write(path: Path, data: bytes) -> None:
    """Atomic journal replacement; same-directory file and directory fsync."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def operation_receipts(journal: Path) -> list[Path]:
    """Select controller-dispatched receipts, never candidate output JSON."""
    return [
        receipt
        for marker in journal.glob("*.dispatch")
        if (receipt := marker.with_suffix(".json")).is_file()
    ]


def script_payload(script: bytes) -> bytes:
    encoded = base64.b64encode(gzip.compress(script, mtime=0)).decode()
    return (
        "#!/usr/bin/env python3\nimport base64,gzip,hashlib\n"
        + f"source = gzip.decompress(base64.b64decode({encoded!r}))\n"
        + f"assert hashlib.sha256(source).hexdigest() == {hashlib.sha256(script).hexdigest()!r}\n"
        + "exec(compile(source, '<cannery-job>', 'exec'))\n"
    ).encode()


class CplRemote:
    def __init__(
        self, client: Path, journal: Path, target: str, *, registry: Path | None = None
    ) -> None:
        self.client, self.journal, self.target = client, journal, target
        self.registry = registry or Path.home() / ".config/cpl/remotes.yaml"
        self._available = False
        self._backend = "ssh"
        self._workspace: PurePosixPath | None = None
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", target):
            raise ValueError("Invalid remote target")
        self.journal.mkdir(mode=0o700, parents=True, exist_ok=True)

    def call(self, *arguments: str, large_output: bool = False) -> dict[str, Any]:
        # Use the installed skill, not a workspace transport implementation.
        (self.client.parent.parent / "SKILL.md").read_text()
        environment = {
            key: value
            for key, value in os.environ.items()
            if key
            in {
                "PATH",
                "HOME",
                "LANG",
                "SSH_AUTH_SOCK",
                "XDG_CONFIG_HOME",
                "XDG_STATE_HOME",
            }
        }
        try:
            output = run(
                [str(self.client), "--config", str(self.registry), "--json", *arguments],
                env=environment,
                timeout=180,
                max_output=1024 * 1024,
                git_limits=False,
                large_output=large_output,
                check=False,
            )
            result = json.loads(output)
            if not isinstance(result, dict):
                raise ValueError("Invalid remote response")
            return result
        except (IntakeError, ValueError, OSError):
            raise RemotePending("Remote observation unavailable") from None

    def preflight(self) -> None:
        if self._available:
            return
        capabilities = self.call("capabilities", self.target)
        if capabilities.get("state") != "available" or not all(
            capabilities.get("capabilities", {}).get(name)
            for name in ("run", "observe", "upload", "download", "cancel")
        ):
            raise RemotePending("Required remote capabilities unavailable")
        backend = capabilities.get("backend")
        if backend not in {"ssh", "remote-agent"}:
            raise ValueError("Unsupported worker transport")
        try:
            config = yaml.safe_load(self.registry.read_text())["targets"][self.target]
            if (
                config.get("backend") != backend
                or backend == "ssh"
                and config.get("transfer", {}).get("mode", "rsync") != "rsync"
            ):
                raise ValueError("Worker registry must match backend; SSH requires rsync")
        except (OSError, KeyError, TypeError, yaml.YAMLError):
            raise ValueError("Worker route registry unavailable") from None
        if backend == "remote-agent":
            workspace = config.get("workspace")
            if not isinstance(workspace, str) or not workspace.startswith("/"):
                raise ValueError("Remote-agent workspace must be absolute")
            self._workspace = PurePosixPath(workspace)
            if ".." in self._workspace.parts:
                raise ValueError("Invalid remote-agent workspace")
        if self.call("preflight", self.target).get("state") != "completed":
            raise RemotePending("Remote preflight unavailable")
        self._available = True
        self._backend = backend

    def operation(
        self,
        name: str,
        script: bytes,
        *,
        workspace: str,
        runtime: str | None,
        npu: int | None = None,
    ) -> str:
        """Submit once, then observe the retained handle across coordinator restarts."""
        if not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", name):
            raise ValueError("Invalid operation name")
        receipt = self.journal / f"{name}.json"
        marker = self.journal / f"{name}.dispatch"
        payload = self.journal / f"{name}.py"
        if not receipt.exists():
            if marker.exists():
                raise RemotePending(
                    "Dispatch receipt pending; never resubmit an ambiguous operation"
                )
            self.preflight()
            cwd = (
                self._agent_path(workspace, allow_root=True)
                if self._backend == "remote-agent"
                else workspace
            )
            # Exclusive creation fences even racing/stale coordinators before side effects.
            try:
                with marker.open("xb") as stream:
                    stream.write(b"dispatching\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            except FileExistsError:
                raise RemotePending("Dispatch already in progress") from None
            directory = os.open(self.journal, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            durable_write(payload.with_suffix(".source.py"), script)
            durable_write(payload, script_payload(script))
            args = [
                "run",
                self.target,
                "--cwd",
                cwd,
                "--file",
                str(payload),
                "--receipt-file",
                str(receipt),
            ]
            if runtime is not None:
                args.extend(["--runtime", runtime])
            if npu is not None:
                args.extend(["--npu", str(npu)])
            self.call(*args)
        try:
            saved = json.loads(receipt.read_bytes())
            handle = saved["handle"]
            if (
                not isinstance(handle, str)
                or not handle.startswith(f"remote:{self.target}:job:")
                or saved.get("target") != self.target
            ):
                raise ValueError("Invalid remote receipt")
        except (OSError, ValueError, KeyError, TypeError):
            raise RemotePending("Dispatch receipt unavailable") from None
        result = self.call("observe", handle)
        state = result.get("state")
        if state == "missing":
            cancelled = self.call("cancel", handle)
            if cancelled.get("state") == "cancelled":
                raise RemoteFailure(125)
        if state == "completed" and result.get("exit") == 0:
            return handle
        if state in {"failed", "cancelled"}:
            raise RemoteFailure(result.get("exit"))
        raise RemotePending("Retained operation is not terminal")

    def transfer(
        self, direction: str, source: str, destination: str, *, large_output: bool = False
    ) -> None:
        """Resume synchronous SSH or one retained remote-agent transfer at a time."""
        if direction not in {"upload", "download"}:
            raise ValueError("Invalid transfer direction")
        self.preflight()
        if large_output and direction != "download":
            raise ValueError("Large evidence budget applies only to downloads")
        if self._backend == "remote-agent":
            self._retained_transfer(direction, source, destination, large_output)
            return
        result = (
            self.call(direction, self.target, source, destination, large_output=True)
            if large_output
            else self.call(direction, self.target, source, destination)
        )
        if result.get("state") != "completed":
            raise RemotePending("Transfer incomplete; resume the same route before dependent work")

    def _agent_path(self, value: str, *, allow_root: bool = False) -> str:
        path = PurePosixPath(value)
        if self._workspace is None or not path.is_absolute() or ".." in path.parts:
            raise ValueError("Remote path must be inside the registered workspace")
        try:
            relative = path.relative_to(self._workspace)
        except ValueError:
            raise ValueError("Remote path is outside the registered workspace") from None
        if not allow_root and relative == PurePosixPath("."):
            raise ValueError("Cannot transfer the registered workspace root")
        return str(relative)

    def _retained_transfer(
        self, direction: str, source: str, destination: str, large_output: bool
    ) -> None:
        relative = self._agent_path(destination if direction == "upload" else source)
        key = (
            "transfer-"
            + hashlib.sha256(json.dumps([direction, source, destination]).encode()).hexdigest()[:24]
        )
        counter = self.journal / (key + ".index")
        index = int(counter.read_text()) if counter.exists() else 0
        if not 0 <= index <= 3:
            raise ValueError("Invalid transfer retry journal")
        if index == 3:
            raise RemoteFailure(125)
        receipt = self.journal / f"{key}-{index}.json"
        marker = receipt.with_suffix(".dispatch")
        if not receipt.exists():
            if marker.exists():
                raise RemotePending("Transfer receipt missing; do not repeat ambiguous dispatch")
            try:
                with marker.open("xb") as stream:
                    stream.write(b"dispatching\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            except FileExistsError:
                raise RemotePending("Transfer dispatch already in progress") from None
            descriptor = os.open(self.journal, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            if direction == "upload":
                self.call(
                    "upload",
                    self.target,
                    source,
                    str(relative),
                    "--receipt-file",
                    str(receipt),
                    "--no-wait",
                )
            else:
                result = self.call(
                    "download", self.target, str(relative), destination, "--timeout", "0"
                )
                durable_write(receipt, json.dumps(result).encode())
        try:
            saved = json.loads(receipt.read_bytes())
            handle = saved["handle"]
            if (
                saved.get("target") != self.target
                or not isinstance(handle, str)
                or not handle.startswith(f"remote:{self.target}:transfer:")
            ):
                raise ValueError("Invalid transfer receipt")
        except (OSError, ValueError, KeyError, TypeError):
            raise RemotePending("Transfer receipt unavailable") from None
        result = self.call("observe", handle)
        if result.get("state") == "missing":
            raise RemotePending("Transfer missing; quarantine without resubmission or cancellation")
        if result.get("state") in {"failed", "cancelled"}:
            # Only confirmed terminal failures may start another transfer. Polling supplies backoff.
            durable_write(counter, str(index + 1).encode())
            raise RemotePending("Transfer failed; retry on next coordinator poll")
        if result.get("state") != "completed":
            raise RemotePending("Retained transfer is not terminal")
        if direction == "download":
            fetched = (
                self.call("fetch", handle, destination, large_output=True)
                if large_output
                else self.call("fetch", handle, destination)
            )
            if fetched.get("state") != "completed":
                raise RemotePending("Retained download not yet materialized")


def settle_operation(
    remote: CplRemote,
    handle: str,
    *,
    cancel_jobs: bool = False,
    fence: Callable[[], None] = lambda: None,
) -> None:
    """Fence cleanup; transfers support observation, never cancellation."""
    result = remote.call("observe", handle)
    if result.get("state") in {"completed", "failed", "cancelled"}:
        return
    if ":transfer:" in handle:
        raise RemotePending("Transfer unresolved; quarantine until retained handle reconciles")
    if result.get("state") == "missing" or cancel_jobs:
        fence()
        result = remote.call("cancel", handle)
        if result.get("state") in {"completed", "failed", "cancelled"}:
            return
    raise RemotePending("Retained operation has not terminated")
