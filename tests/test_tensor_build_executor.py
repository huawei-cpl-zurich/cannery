import io
import json
import tarfile
import zipfile
from dataclasses import replace
from uuid import uuid4

import pytest
from test_build_executor import setup_executor as setup_executor
from test_build_sandbox import catlass_pins
from test_tensor_spec_reader import tensor_files
from test_tensor_submission_reader import tensor_files as submission_files

from cannery.artifacts import digest
from cannery.builds.executor import ExecutionFailure
from cannery.builds.remote import RemotePending
from cannery.builds.sandbox import CatlassToolchain
from cannery.builds.tensor_signatures import tensor_variants
from cannery.git_intake import GitSource
from cannery.git_intake.snapshots import SelectedTree
from cannery.registry.tensor_manifest import validate_tensor_spec
from cannery.submissions.manifest import validate_tensor_submission


@pytest.fixture
def tensor_build(setup_executor):
    executor, candidate, remote, memory = setup_executor
    files = submission_files("catlass")
    valid = validate_tensor_submission(files, "candidate")
    binding = executor.snapshots.commit(
        "d" * 64,
        GitSource(url="https://example.invalid/tensor.git", commit="b" * 40, path="candidate"),
        SelectedTree(files, dict.fromkeys(files, "100644"), "c" * 40),
    )
    candidate.snapshot = binding.model_dump(mode="json")
    candidate.manifest = valid.manifest.model_dump(mode="json")
    candidate.content_sha256, candidate.file_sha256 = valid.content_sha256, valid.file_sha256
    pins = catlass_pins() | {"package_sha256": "a" * 64, "aic_bitcode_sha256": "b" * 64}
    executor.profile = executor.profile.model_copy(update={"catlass": CatlassToolchain(**pins)})
    with zipfile.ZipFile(executor.wheel, "w") as archive:
        archive.writestr("cannery_kernels/tensor_abi.h", "int cannery_prepare_tensor_v2(void);")
    executor.wheel_sha256 = digest(executor.wheel.read_bytes())
    return executor, candidate, remote, memory, validate_tensor_spec(tensor_files(), "spec")


def test_tensor_compile_package_and_native_smoke_follow_durable_flow(tensor_build):
    executor, candidate, remote, memory, frozen = tensor_build
    attempt = uuid4()
    result = executor.execute(candidate, attempt, lambda: None, frozen=frozen)
    with tarfile.open(fileobj=io.BytesIO(memory.get(result))) as archive:
        metadata = json.load(archive.extractfile("metadata.json"))
    assert metadata["schema_version"] == 3 and metadata["abi"] == "tensor-v2"
    assert metadata["variants"] == tensor_variants(frozen)
    compiler = json.loads(metadata["compiler"])
    assert compiler["catlass"]["package_sha256"] == "a" * 64
    assert "bitcode_root" not in compiler["catlass"]
    context = json.loads((remote.journal / "context.json").read_bytes())
    assert context["tensor_spec"] == frozen.spec.content_sha256
    assert remote.started == {"prepare", "unpack-source", "compile", "unpack-runtime", "runtime-0"}
    assert executor.execute(candidate, attempt, lambda: None, frozen=frozen) == result
    assert remote.uploads.count("source.tar.gz") == 1


@pytest.mark.parametrize("failure", ["compile", "runtime-0", "busy"])
def test_tensor_lost_reply_and_device_busy_resume_same_build(tensor_build, failure):
    executor, candidate, remote, memory, frozen = tensor_build
    remote.failure = failure
    attempt = uuid4()
    with pytest.raises(RemotePending):
        executor.execute(candidate, attempt, lambda: None, frozen=frozen)
    assert not (remote.journal / "admission.json").exists()
    remote.failure = None
    executor.execute(candidate, attempt, lambda: None, frozen=frozen)
    assert remote.uploads.count("source.tar.gz") == 1


def test_tensor_spec_identity_change_fences_retained_build(tensor_build):
    executor, candidate, remote, memory, frozen = tensor_build
    remote.failure = "compile"
    attempt = uuid4()
    with pytest.raises(RemotePending):
        executor.execute(candidate, attempt, lambda: None, frozen=frozen)
    changed = replace(frozen, spec=replace(frozen.spec, content_sha256="f" * 64))
    with pytest.raises(ExecutionFailure, match="worker_profile_changed"):
        executor.execute(candidate, attempt, lambda: None, frozen=changed)


def test_tensor_requires_pinned_header_before_remote_dispatch(tensor_build):
    executor, candidate, remote, memory, frozen = tensor_build
    with zipfile.ZipFile(executor.wheel, "w") as archive:
        archive.writestr("other", "missing tensor ABI")
    executor.wheel_sha256 = digest(executor.wheel.read_bytes())
    with pytest.raises(ExecutionFailure, match="tensor_loader_header_missing"):
        executor.execute(candidate, uuid4(), lambda: None, frozen=frozen)
    assert not remote.started and not memory.puts
