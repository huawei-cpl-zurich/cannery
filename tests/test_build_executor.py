import io
import json
import tarfile
from pathlib import Path
from uuid import uuid4

import pytest
import yaml
from submission_fixtures import submission_files
from test_build_sandbox import profile

from cannery.artifacts import ArtifactRef, digest
from cannery.builds.executor import BuildExecutor, ExecutionFailure, binary_package
from cannery.builds.remote import RemoteFailure, RemotePending
from cannery.git_intake import GitSource
from cannery.git_intake.snapshots import SelectedTree, SnapshotStore
from cannery.registry.freeze import kernel_key
from cannery.submissions.manifest import validate_submission
from cannery.submissions.models import Candidate


def elf():
    data = bytearray(128)
    data[:6] = b"\x7fELF\x02\x01"
    data[16:18] = (3).to_bytes(2, "little")
    data[18:20] = (183).to_bytes(2, "little")
    return bytes(data)


class Memory:
    def __init__(self):
        self.data, self.puts = {}, 0

    def put(self, path, data):
        self.puts += 1
        checksum = digest(data)
        self.data[checksum] = data
        return ArtifactRef(repository="private", revision=checksum, sha256=checksum, path=path)

    def get(self, ref):
        return self.data[ref.sha256]


class Remote:
    def __init__(self, root, failure=None):
        self.journal = root
        root.mkdir()
        self.started, self.uploads, self.failure = set(), [], failure

    def operation(self, name, script, **kwargs):
        fresh = name not in self.started
        self.started.add(name)
        handle = "remote:test:job:" + name
        (self.journal / f"{name}.dispatch").write_text("submitted")
        (self.journal / f"{name}.json").write_text(json.dumps({"handle": handle}))
        if fresh and name == self.failure:
            raise RemotePending("lost response")
        if fresh and name == "cleanup-0" and self.failure == "cleanup":
            raise RemoteFailure(1)
        if name == "runtime-0" and self.failure == "busy":
            raise RemoteFailure(75)
        return handle

    def call(self, *args):
        if args[0] == "logs":
            return {"content": digest(elf()), "state": "completed"}
        return {"state": "completed", "exit": 0}

    def transfer(self, kind, source, destination):
        if kind == "upload":
            self.uploads.append(Path(source).name)
        else:
            Path(destination).write_bytes(elf())


@pytest.fixture
def setup_executor(tmp_path):
    files = submission_files("ascendc")
    manifest = yaml.safe_load(files["candidate/submission.yaml"])
    manifest["implementation"]["device"] = "a5"
    manifest["implementation"]["build"]["abi"] = "unary-f32-v1"
    files["candidate/submission.yaml"] = yaml.safe_dump(manifest).encode()
    source = GitSource(url="https://example.invalid/repo.git", commit="a" * 40, path="candidate")
    snapshots = SnapshotStore(tmp_path / "snapshots")
    binding = snapshots.commit(
        kernel_key("team", "identity"),
        source,
        SelectedTree(files, dict.fromkeys(files, "100644"), "b" * 40),
    )
    valid = validate_submission(files, source.path)
    key = uuid4()
    candidate = Candidate(
        id=key,
        binding_id=key,
        binding={},
        snapshot=binding.model_dump(mode="json"),
        manifest=valid.manifest.model_dump(mode="json"),
        file_sha256=valid.file_sha256,
        content_sha256=valid.content_sha256,
    )
    wheel = tmp_path / "loader-0.1-py3-none-any.whl"
    wheel.write_bytes(b"CPU fixture; never imported")
    remote, memory = Remote(tmp_path / "journal"), Memory()
    executor = BuildExecutor(
        profile(tmp_path / "worker"), remote, snapshots, memory, wheel, digest(wheel.read_bytes())
    )
    return executor, candidate, remote, memory


def test_retained_git_build_private_roundtrip_and_source_free_package(setup_executor):
    executor, candidate, remote, memory = setup_executor
    result = executor.execute(candidate, uuid4(), lambda: None)
    assert result.repository == "private"
    with tarfile.open(fileobj=io.BytesIO(memory.get(result)), mode="r:") as archive:
        assert set(archive.getnames()) == {"metadata.json", "kernel.so"}
        metadata = json.load(archive.extractfile("metadata.json"))
        assert metadata["source_sha256"] == candidate.content_sha256
    assert remote.started == {"prepare", "unpack-source", "compile", "unpack-runtime", "runtime-0"}


def test_disconnect_recovers_attempt_without_reuploading_source(setup_executor):
    executor, candidate, remote, memory = setup_executor
    remote.failure = "compile"
    attempt = uuid4()
    with pytest.raises(RemotePending):
        executor.execute(candidate, attempt, lambda: None)
    assert not memory.puts and "runtime-0" not in remote.started
    executor.execute(candidate, attempt, lambda: None)
    assert remote.uploads.count("source.tar.gz") == 1


def test_busy_device_keeps_compilation_and_uses_new_health_probe(setup_executor):
    executor, candidate, remote, memory = setup_executor
    remote.failure = "busy"
    attempt = uuid4()
    with pytest.raises(RemotePending):
        executor.execute(candidate, attempt, lambda: None)
    executor.execute(candidate, attempt, lambda: None)
    assert {"runtime-0", "runtime-1"} <= remote.started
    assert remote.uploads.count("source.tar.gz") == 1


def test_tampered_sources_loader_and_changed_profile_never_dispatch(setup_executor):
    executor, candidate, remote, memory = setup_executor
    original = candidate.content_sha256
    candidate.content_sha256 = "f" * 64
    with pytest.raises(ExecutionFailure, match="retained_source_mismatch"):
        executor.execute(candidate, uuid4(), lambda: None)
    candidate.content_sha256 = original
    executor.wheel_sha256 = "f" * 64
    with pytest.raises(ExecutionFailure, match="loader_digest_mismatch"):
        executor.execute(candidate, uuid4(), lambda: None)
    assert not remote.started and not memory.puts


def test_binary_package_is_deterministic_and_rejects_wrong_architecture(tmp_path):
    config = profile(tmp_path)
    assert binary_package(elf(), "a" * 64, config) == binary_package(elf(), "a" * 64, config)
    wrong = bytearray(elf())
    wrong[18:20] = (62).to_bytes(2, "little")
    with pytest.raises(ExecutionFailure):
        binary_package(bytes(wrong), "a" * 64, config)


def test_admission_checkpoint_survives_database_failure_after_scratch_cleanup(setup_executor):
    executor, candidate, remote, memory = setup_executor
    attempt = uuid4()
    admitted = executor.execute(candidate, attempt, lambda: None)
    executor.cleanup(attempt)

    def unavailable(*args):
        raise AssertionError("must not repeat transfers after successful admission")

    remote.transfer = unavailable
    assert executor.execute(candidate, attempt, lambda: None) == admitted


def test_worker_profile_change_cannot_reuse_an_old_attempt(setup_executor):
    executor, candidate, remote, memory = setup_executor
    attempt = uuid4()
    remote.failure = "compile"
    with pytest.raises(RemotePending):
        executor.execute(candidate, attempt, lambda: None)
    started = remote.started.copy()
    executor.profile = executor.profile.model_copy(update={"compiler": "different"})
    with pytest.raises(ExecutionFailure, match="worker_profile_changed"):
        executor.execute(candidate, attempt, lambda: None)
    assert remote.started == started and not memory.puts


def test_cleanup_retries_only_after_terminal_failed_operation(setup_executor):
    executor, candidate, remote, memory = setup_executor
    attempt = uuid4()
    admitted = executor.execute(candidate, attempt, lambda: None)
    remote.failure = "cleanup"
    with pytest.raises(RemotePending, match="retain ownership"):
        executor.cleanup(attempt)
    assert (remote.journal / "cleanup-index").read_text() == "1"
    assert executor.execute(candidate, attempt, lambda: None) == admitted
    executor.cleanup(attempt)
    assert {"cleanup-0", "cleanup-1"} <= remote.started


def test_lease_loss_during_transfer_cannot_dispatch_next_remote_stage(setup_executor):
    executor, candidate, remote, memory = setup_executor
    lost = []
    original = remote.transfer

    def transfer(*args):
        original(*args)
        lost.append(True)

    remote.transfer = transfer

    def fence():
        if lost:
            raise RemotePending("stale lease")

    with pytest.raises(RemotePending, match="stale lease"):
        executor.execute(candidate, uuid4(), fence)
    assert remote.started == {"prepare"}
