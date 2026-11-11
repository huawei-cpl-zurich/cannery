import io
import json
import tarfile
from uuid import uuid4

import pytest
import yaml
from sqlalchemy import delete, select
from sqlalchemy.orm import Session
from test_build_executor import elf
from test_build_executor import setup_executor as setup_executor
from test_build_sandbox import catlass_pins
from test_catlass_contract import native_manifest
from test_submission_requests import initial
from test_submission_requests import records as records

from cannery.builds.coordinator import discover
from cannery.builds.executor import ExecutionFailure, binary_package
from cannery.builds.remote import RemotePending
from cannery.builds.sandbox import WorkerProfile
from cannery.builds.state import BuildState, claim_build, queue_build
from cannery.git_intake import GitSource
from cannery.git_intake.snapshots import SelectedTree
from cannery.registry.freeze import kernel_key
from cannery.submissions.manifest import validate_submission
from cannery.submissions.models import Candidate


def native_candidate(executor):
    files, manifest = native_manifest()
    files["candidate/submission.yaml"] = yaml.safe_dump(manifest).encode()
    source = GitSource(url="https://example.invalid/catlass.git", commit="a" * 40, path="candidate")
    binding = executor.snapshots.commit(
        kernel_key("team", "catlass"),
        source,
        SelectedTree(files, dict.fromkeys(files, "100644"), "b" * 40),
    )
    valid = validate_submission(files, "candidate")
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
    executor.profile = WorkerProfile.model_validate(
        executor.profile.model_dump() | {"catlass": catlass_pins()}
    )
    return candidate, valid.manifest


def test_catlass_retained_source_private_package_and_all_declared_shapes(setup_executor):
    executor, _, remote, memory = setup_executor
    candidate, manifest = native_candidate(executor)
    result = executor.execute(candidate, uuid4(), lambda: None)
    with tarfile.open(fileobj=io.BytesIO(memory.get(result))) as tar:
        assert set(tar.getnames()) == {"metadata.json", "kernel.so"}
        metadata = json.load(tar.extractfile("metadata.json"))
    assert metadata["schema_version"] == 2 and metadata["abi"] == "catlass-unary-f32-v1"
    assert (
        metadata["specializations"]
        == manifest.implementation.build.model_dump(mode="json")["specializations"]
    )
    assert "revision" in json.loads(metadata["compiler"])["catlass"]
    assert "bitcode_root" not in metadata["compiler"]
    with tarfile.open(remote.journal / "source.tar.gz") as tar:
        plan = json.load(tar.extractfile("build.json"))
    assert plan["wrapper_name"] == "launch.py" and plan["command"] == ["python", "launch.py"]
    assert plan["catlass"] == catlass_pins()
    assert "torch" not in plan["wrapper"] and "import cannery" not in plan["wrapper"]
    assert remote.started == {"prepare", "unpack-source", "compile", "unpack-runtime", "runtime-0"}


def test_catlass_missing_pins_fails_before_any_dispatch(setup_executor):
    executor, _, remote, memory = setup_executor
    candidate, _ = native_candidate(executor)
    executor.profile = executor.profile.model_copy(update={"catlass": None})
    with pytest.raises(ExecutionFailure, match="catlass_toolchain_unavailable"):
        executor.execute(candidate, uuid4(), lambda: None)
    assert not remote.started and not memory.puts


@pytest.mark.parametrize("failure", ["compile", "busy"])
def test_catlass_network_or_device_failure_resumes_same_source_and_attempt(setup_executor, failure):
    executor, _, remote, memory = setup_executor
    candidate, _ = native_candidate(executor)
    remote.failure = failure
    attempt = uuid4()
    with pytest.raises(RemotePending):
        executor.execute(candidate, attempt, lambda: None)
    reference = executor.execute(candidate, attempt, lambda: None)
    assert remote.uploads.count("source.tar.gz") == 1
    assert memory.get(reference)


def test_catlass_pin_change_fences_recovered_attempt(setup_executor):
    executor, _, remote, memory = setup_executor
    candidate, _ = native_candidate(executor)
    remote.failure = "compile"
    attempt = uuid4()
    with pytest.raises(RemotePending):
        executor.execute(candidate, attempt, lambda: None)
    pins = executor.profile.catlass.model_copy(update={"hivmc_sha256": "0" * 64})
    executor.profile = executor.profile.model_copy(update={"catlass": pins})
    with pytest.raises(ExecutionFailure, match="worker_profile_changed"):
        executor.execute(candidate, attempt, lambda: None)
    assert not memory.puts


def test_schema_one_package_bytes_stay_identical_when_manifest_is_supplied(setup_executor):
    executor, candidate, _, _ = setup_executor
    from cannery.submissions.manifest import SubmissionManifest

    manifest = SubmissionManifest.model_validate(candidate.manifest)
    assert binary_package(elf(), "a" * 64, executor.profile) == binary_package(
        elf(), "a" * 64, executor.profile, manifest
    )


def test_catlass_discovery_queues_once(records):
    engine, _ = records
    key = initial(records)
    _, manifest = native_manifest()
    with Session(engine) as session, session.begin():
        candidate = Candidate(
            id=key,
            binding_id=key,
            binding={},
            manifest=manifest,
            snapshot={},
            content_sha256="a" * 64,
            file_sha256={},
        )
        session.add(candidate)
        session.flush()
        assert queue_build(session, candidate)
    with engine.begin() as connection:
        connection.execute(delete(BuildState).where(BuildState.id == key))
    discover(engine)
    discover(engine)
    with Session(engine) as session:
        assert list(session.scalars(select(BuildState.id))) == [key]
    assert claim_build(engine, key) is not None


def test_catlass_empty_only_contract_is_not_queued_and_discovery_records_failure(records):
    engine, _ = records
    key = initial(records)
    _, manifest = native_manifest()
    manifest["implementation"]["build"]["specializations"] = [{"shape": [0], "dtype": "float32"}]
    with Session(engine) as session, session.begin():
        candidate = Candidate(
            id=key,
            binding_id=key,
            binding={},
            manifest=manifest,
            snapshot={},
            content_sha256="a" * 64,
            file_sha256={},
        )
        session.add(candidate)
        session.flush()
        assert not queue_build(session, candidate)
    discover(engine)
    with Session(engine) as session:
        state = session.get(BuildState, key)
        assert state.status == "failed" and state.error == "unsupported_build_contract"
    assert claim_build(engine, key) is None
