import json

import pytest
from sqlalchemy import update
from sqlalchemy.orm import Session
from submission_fixtures import submission_files
from test_build_executor import elf
from test_build_executor import setup_executor as setup_executor
from test_build_sandbox import profile, setup, supervise
from test_build_state import candidate
from test_submission_requests import records as records

from cannery.builds.ascendc import AscendCAdapter
from cannery.builds.coordinator import discover
from cannery.builds.executor import ExecutionFailure, binary_package
from cannery.builds.sandbox import WorkerProfile
from cannery.builds.state import BuildState, claim_build, due_builds
from cannery.submissions.manifest import SubmissionManifest
from cannery.submissions.models import Candidate


def a3_profile(tmp_path):
    config = profile(tmp_path / "worker")
    raw = config.model_dump()
    raw["device"] = "a3"
    raw["fingerprint"]["soc"] = "Ascend910B4-1"
    return WorkerProfile.model_validate(raw)


def test_profiles_preserve_a5_contexts_and_reject_cross_device(tmp_path):
    config = profile(tmp_path / "worker")
    assert "device" not in config.model_dump(mode="json")
    assert "toolchain_in_image" not in config.model_dump(mode="json")
    assert WorkerProfile.model_validate_json(config.model_dump_json()) == config
    a3 = a3_profile(tmp_path)
    assert a3.model_dump()["device"] == "a3"
    for changes in [{"device": "a5"}, {"fingerprint": config.fingerprint}]:
        with pytest.raises(ValueError):
            WorkerProfile.model_validate(a3.model_dump() | changes)


def test_native_adapter_and_package_use_matching_device(tmp_path):
    import io
    import tarfile

    import yaml

    declaration = yaml.safe_load(submission_files("ascendc")["candidate/submission.yaml"])
    declaration["implementation"]["device"] = "a3"
    declaration["implementation"]["build"]["abi"] = "unary-f32-v1"
    manifest = SubmissionManifest.model_validate(declaration)
    assert "--npu-arch=dav-2201" in AscendCAdapter().build(manifest).command
    raw = binary_package(elf(), "a" * 64, a3_profile(tmp_path), manifest)
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        metadata = json.load(archive.extractfile("metadata.json"))
    assert metadata["device"] == "a3" and metadata["schema_version"] == 1
    with pytest.raises(ExecutionFailure, match="worker_device_mismatch"):
        binary_package(elf(), "a" * 64, profile(tmp_path / "worker"), manifest)


def test_a5_cannot_discover_or_claim_an_a3_build(records):
    engine, _ = records
    key = candidate(records)
    with engine.begin() as connection:
        with Session(engine) as db:
            manifest = dict(db.get(Candidate, key).manifest)
        manifest["implementation"] = manifest["implementation"] | {"device": "a3"}
        connection.execute(update(Candidate).where(Candidate.id == key).values(manifest=manifest))
        connection.execute(BuildState.__table__.delete().where(BuildState.id == key))
    discover(engine)
    with Session(engine) as db:
        assert db.get(BuildState, key) is None
    discover(engine, device="a3")
    assert str(key) not in due_builds(engine) and str(key) in due_builds(engine, device="a3")
    assert claim_build(engine, key) is None
    assert claim_build(engine, key, device="a3") is not None


def test_wrong_device_executor_never_dispatches_remote_work(setup_executor, tmp_path):
    executor, frozen, remote, _ = setup_executor
    executor.profile = a3_profile(tmp_path)
    from uuid import uuid4

    with pytest.raises(ExecutionFailure, match="worker_device_mismatch"):
        executor.execute(frozen, uuid4(), lambda: None)
    assert not remote.started


def test_image_pinned_toolchain_needs_no_host_sdk_or_inherited_environment(tmp_path, monkeypatch):
    config, token, calls = setup(tmp_path, monkeypatch)
    config = WorkerProfile.model_validate(config.model_dump() | {"toolchain_in_image": True})
    monkeypatch.delenv("ASCEND_HOME_PATH")
    supervise(config, token, "compile")
    args = next(call for call in calls if call[:2] == ["docker", "run"])
    assert "--device" not in args and "PATH" not in args and "ASCEND_HOME_PATH" not in args
    mounts = [args[i + 1] for i, arg in enumerate(args) if arg == "--mount"]
    assert len(mounts) == 1 and "dst=/input,readonly" in mounts[0]


def test_old_supervisor_checkpoint_cannot_silently_resume_under_new_code(setup_executor):
    from uuid import uuid4

    executor, frozen, remote, memory = setup_executor
    attempt = uuid4()
    executor.execute(frozen, attempt, lambda: None)
    context = remote.journal / "context.json"
    old = json.loads(context.read_bytes()) | {"job_contract": "b" * 64}
    context.write_text(json.dumps(old, sort_keys=True))
    before, uploads, puts = context.read_bytes(), list(remote.uploads), memory.puts
    with pytest.raises(ExecutionFailure, match="worker_profile_changed"):
        executor.execute(frozen, attempt, lambda: None)
    assert context.read_bytes() == before and remote.uploads == uploads and memory.puts == puts


def test_numerical_discovery_and_claims_are_device_scoped(records):
    from cannery.numerics.coordinator import discover as discover_numerics
    from cannery.numerics.state import NumericalState, claim_numerical, due_numericals

    engine, _ = records
    a3, a5 = candidate(records), candidate(records)
    with Session(engine) as db, db.begin():
        manifest = db.get(Candidate, a3).manifest
        db.execute(
            update(Candidate)
            .where(Candidate.id == a3)
            .values(
                manifest=manifest
                | {"implementation": manifest["implementation"] | {"device": "a3"}}
            )
        )
        for key in (a3, a5):
            db.get(BuildState, key).status = "awaiting_validation"
    discover_numerics(engine)
    with Session(engine) as db:
        assert db.get(NumericalState, a3) is None
        assert db.get(NumericalState, a5).status == "blocked"
    discover_numerics(engine, device="a3")
    with Session(engine) as db, db.begin():
        for key in (a3, a5):
            row = db.get(NumericalState, key)
            row.status, row.error = "queued", None
    assert due_numericals(engine) == [str(a5)]
    assert due_numericals(engine, device="a3") == [str(a3)]
    assert claim_numerical(engine, a3) is None
    assert claim_numerical(engine, a5, device="a3") is None
    a3_claim = claim_numerical(engine, a3, device="a3")
    a5_claim = claim_numerical(engine, a5)
    assert a3_claim is not None and a5_claim is not None
    assert claim_numerical(engine, a3) is None
    assert claim_numerical(engine, a5, device="a3") is None
    with Session(engine) as db:
        assert all(db.get(NumericalState, key).attempts == 1 for key in (a3, a5))
