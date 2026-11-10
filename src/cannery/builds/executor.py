"""Git snapshot -> isolated build -> private staging -> isolated loadability smoke."""

import io
import json
import re
import tarfile
from collections.abc import Callable
from pathlib import Path
from uuid import UUID

from cannery.artifacts import ArtifactRef, ArtifactStore, digest
from cannery.builds.ascendc import AscendCAdapter, BuildError
from cannery.builds.remote import CplRemote, RemoteFailure, RemotePending, durable_write
from cannery.builds.sandbox import WorkerProfile, job_script
from cannery.git_intake import SnapshotBinding
from cannery.git_intake.snapshots import SnapshotStore
from cannery.submissions.manifest import validate_submission
from cannery.submissions.models import Candidate


class ExecutionFailure(BuildError):
    def __init__(self, code: str, retryable: bool = False) -> None:
        self.code, self.retryable = code, retryable
        super().__init__(code)


def binary_package(binary: bytes, source: str, profile: WorkerProfile) -> bytes:
    if (
        not 64 <= len(binary) <= 63 * 1024 * 1024
        or binary[:6] != b"\x7fELF\x02\x01"
        or int.from_bytes(binary[16:18], "little") != 3
        or int.from_bytes(binary[18:20], "little") != 183
        or not re.fullmatch(r"[a-f0-9]{64}", source)
    ):
        raise ExecutionFailure("invalid_compiled_binary")
    metadata = {
        "schema_version": 1,
        "abi": "unary-f32-v1",
        "device": "a5",
        "source_sha256": source,
        "library_sha256": digest(binary),
        "runtime": profile.fingerprint,
        "compiler": profile.compiler,
    }
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    if len(encoded) > 16384:
        raise ExecutionFailure("invalid_worker_profile")
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name, value in [("metadata.json", encoded), ("kernel.so", binary)]:
            item = tarfile.TarInfo(name)
            item.size, item.mode = len(value), 0o400
            archive.addfile(item, io.BytesIO(value))
    return output.getvalue()


def unpack_script(profile: WorkerProfile, attempt: UUID, label: str, checksum: str) -> bytes:
    # Names and digests come from the trusted coordinator, never author commands.
    root = str(Path(profile.workspace) / attempt.hex)
    return f"""#!/usr/bin/env python3
import hashlib,tarfile
from pathlib import Path
root = Path({root!r})
bundle = root / {label + ".tar.gz"!r}
assert hashlib.sha256(bundle.read_bytes()).hexdigest() == {checksum!r}
size = 0
with tarfile.open(bundle, 'r:gz') as archive:
    count = 0
    for member in archive:
        count += 1
        path = Path(member.name)
        assert (count <= 4096 and member.isfile() and not path.is_absolute()
                and '..' not in path.parts and 0 <= member.size <= 64*1024*1024)
        size += member.size
        assert size <= 128*1024*1024
        target = root / 'input' / path
        target.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
        for directory in [target.parent, *target.parent.parents]:
            if directory == root / 'input' or directory.is_relative_to(root / 'input'):
                directory.chmod(0o755)
            else:
                break
        target.write_bytes(archive.extractfile(member).read())
        target.chmod(0o444)
""".encode()


class BuildExecutor:
    def __init__(
        self,
        profile: WorkerProfile,
        remote: CplRemote,
        snapshots: SnapshotStore,
        artifacts: ArtifactStore,
        wheel: Path,
        wheel_sha256: str,
    ) -> None:
        self.profile, self.remote, self.snapshots = profile, remote, snapshots
        self.artifacts, self.wheel, self.wheel_sha256 = artifacts, wheel, wheel_sha256

    def operation(self, name: str, script: bytes, *, npu: int | None = None) -> str:
        return self.remote.operation(
            name, script, workspace=self.profile.workspace, runtime=self.profile.runtime, npu=npu
        )

    def upload(self, attempt: UUID, label: str, files: dict[str, bytes]) -> None:
        bundle = self.remote.journal / f"{label}.tar.gz"
        if not (self.remote.journal / f"unpack-{label}.dispatch").exists():
            with tarfile.open(bundle, "w:gz") as archive:
                for name, value in sorted(files.items()):
                    item = tarfile.TarInfo(name)
                    item.size, item.mode = len(value), 0o444
                    archive.addfile(item, io.BytesIO(value))
            destination = str(Path(self.profile.workspace) / attempt.hex / bundle.name)
            self.remote.transfer("upload", str(bundle), destination)
        self.operation(
            "unpack-" + label,
            unpack_script(self.profile, attempt, label, digest(bundle.read_bytes())),
        )

    def execute(
        self, candidate: Candidate, attempt: UUID, fence: Callable[[], None]
    ) -> ArtifactRef:
        binding = SnapshotBinding.model_validate(candidate.snapshot)
        files = self.snapshots.read(binding).files
        validated = validate_submission(files, binding.source.path)
        if (
            validated.content_sha256 != candidate.content_sha256
            or validated.file_sha256 != candidate.file_sha256
            or validated.manifest.model_dump(mode="json") != candidate.manifest
        ):
            raise ExecutionFailure("retained_source_mismatch")
        plan = AscendCAdapter().build(validated.manifest)
        if self.wheel.stat().st_size > 2 * 1024 * 1024:
            raise ExecutionFailure("loader_size_limit")
        wheel = self.wheel.read_bytes()
        if digest(wheel) != self.wheel_sha256 or not re.fullmatch(
            r"[A-Za-z0-9_.-]+\.whl", self.wheel.name
        ):
            raise ExecutionFailure("loader_digest_mismatch")
        context = json.dumps(
            {
                "profile": self.profile.model_dump(mode="json"),
                "source": candidate.content_sha256,
                "wheel": self.wheel_sha256,
                "wrapper": plan.wrapper,
                "command": plan.command,
                "job_contract": digest(job_script(self.profile, attempt, "runtime")),
            },
            sort_keys=True,
        ).encode()
        context_path = self.remote.journal / "context.json"
        if context_path.exists() and context_path.read_bytes() != context:
            raise ExecutionFailure("worker_profile_changed")
        durable_write(context_path, context)
        admission = self.remote.journal / "admission.json"
        if admission.exists():
            reference = ArtifactRef.model_validate_json(admission.read_bytes())
            self.artifacts.get(reference)
            return reference
        fence()
        self.operation("prepare", job_script(self.profile, attempt, "prepare"))
        inputs = {"tree/" + name: value for name, value in files.items()}
        inputs["build.json"] = json.dumps(
            {
                "path": binding.source.path,
                "wrapper": plan.wrapper,
                "command": plan.command,
                "compiler": self.profile.compiler,
            }
        ).encode()
        inputs[self.wheel.name] = wheel
        fence()
        self.upload(attempt, "source", inputs)
        fence()
        try:
            handle = self.operation("compile", job_script(self.profile, attempt, "compile"))
        except RemoteFailure:
            raise ExecutionFailure("compile_failed") from None
        # Trusted host supervision hashes the complete output after container exit.
        logs = self.remote.call("logs", handle, "--tail", "2")
        checksum = logs.get("content", "").strip()
        if not re.fullmatch(r"[a-f0-9]{64}", checksum):
            raise ExecutionFailure("compiler_evidence_missing")
        library = self.remote.journal / "kernel.so"
        fence()
        self.remote.transfer(
            "download",
            str(Path(self.profile.workspace) / attempt.hex / "output/kernel.so"),
            str(library),
        )
        if library.is_symlink() or library.stat().st_size > 63 * 1024 * 1024:
            raise ExecutionFailure("invalid_compiled_binary")
        binary = library.read_bytes()
        if digest(binary) != checksum:
            raise ExecutionFailure("compiled_digest_mismatch")
        package = binary_package(binary, candidate.content_sha256, self.profile)
        fence()
        reference = self.artifacts.put("builds/" + attempt.hex + "/package.tar", package)
        downloaded = self.artifacts.get(reference)
        if downloaded != package:
            raise ExecutionFailure("artifact_readback_mismatch")
        fence()
        self.upload(
            attempt,
            "runtime",
            {
                "admitted.tar": downloaded,
                "artifact.json": reference.model_dump_json().encode(),
                self.wheel.name: wheel,
            },
        )
        index_path = self.remote.journal / "runtime-index"
        index = int(index_path.read_text()) if index_path.exists() else 0
        if index >= 360:
            raise ExecutionFailure("device_unavailable")
        fence()
        try:
            self.operation(
                f"runtime-{index}",
                job_script(self.profile, attempt, "runtime"),
                npu=self.profile.npu,
            )
        except RemoteFailure as error:
            if error.exit_code == 75:
                durable_write(index_path, str(index + 1).encode())
                raise RemotePending("Waiting for a healthy idle device") from None
            raise ExecutionFailure("runtime_failed", retryable=True) from None
        durable_write(admission, reference.model_dump_json().encode())
        return reference

    def cleanup(self, attempt: UUID) -> None:
        context = self.remote.journal / "context.json"
        if context.exists():
            self.profile = WorkerProfile.model_validate(json.loads(context.read_bytes())["profile"])
            self.remote.target = self.profile.target
        # Do not race a retained operation after a controller/transport interruption.
        for marker in self.remote.journal.glob("*.dispatch"):
            receipt = marker.with_suffix(".json")
            if not receipt.exists():
                raise RemotePending("Ambiguous operation requires reconciliation")
            handle = json.loads(receipt.read_text())["handle"]
            result = self.remote.call("observe", handle)
            if result.get("state") == "missing":
                result = self.remote.call("cancel", handle)
            if result.get("state") not in {"completed", "failed", "cancelled"}:
                raise RemotePending("Retained operation has not terminated")
        self.operation("cleanup", job_script(self.profile, attempt, "cleanup"))
