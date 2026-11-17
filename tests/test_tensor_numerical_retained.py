from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session
from test_baseline_ingress import ingress as ingress
from test_build_coordinator import Executor
from test_build_state import candidate

from cannery.artifacts import ArtifactError
from cannery.builds.executor import ExecutionFailure
from cannery.builds.state import claim_build, finish_build
from cannery.git_intake import SnapshotBinding
from cannery.git_intake.snapshots import SnapshotStore
from cannery.numerics import coordinator
from cannery.numerics.baseline_state import BaselineState
from cannery.numerics.baselines import BaselinePending, BaselineUnavailable
from cannery.registry.manifest import validate_spec
from cannery.registry.models import Kernel, SpecVersion
from cannery.registry.tensor_manifest import validate_tensor_spec


@pytest.fixture
def retained(ingress, tmp_path):
    advance, engine, cache_key, control, memory, reference = ingress
    evidence = advance()
    with Session(engine) as db:
        binding = SnapshotBinding.model_validate(db.get(BaselineState, cache_key).snapshot)
    snapshots = SnapshotStore(tmp_path / "snapshots")
    files = snapshots.read(binding).files
    tensor = control["generator"] == "sha256-tensor-v2"
    frozen = validate_tensor_spec(files, "spec").spec if tensor else validate_spec(files, "spec")
    spec_id = uuid4()
    with Session(engine) as db, db.begin():
        db.add(Kernel(id="kernel", namespace="team", name="identity"))
        db.flush()
        db.add(
            SpecVersion(
                id=spec_id,
                kernel_id="kernel",
                version=1,
                content_sha256=frozen.content_sha256,
                manifest=frozen.manifest.model_dump(mode="json"),
                snapshot=binding.model_dump(mode="json"),
                file_sha256=frozen.file_sha256,
                smoke=evidence.model_dump(mode="json"),
            )
        )
    key = candidate((engine, spec_id))
    claim = claim_build(engine, key)
    assert finish_build(engine, claim, artifact=reference)
    executor = Executor(tmp_path / "retained")
    executor.snapshots, executor.artifacts = snapshots, memory
    return engine, key, executor, control, tensor, files


def test_retained_reuses_exact_cached_oracle_and_contract(retained):
    engine, key, executor, control, tensor, _ = retained
    coordinator.discover(engine)
    first = coordinator.retained(engine, key, executor)
    second = coordinator.retained(engine, key, executor)
    assert first["baseline"] == second["baseline"]
    assert first["spec"].manifest.schema_version == (2 if tensor else 1)
    assert first["suite"].generator == control["generator"]
    assert ("contract" in first) == tensor
    if tensor:
        assert first["contract"] == control["contract"]
        assert set(first["baseline"].outputs["small"]) == {"updated", "indices"}
    assert control["calls"] == 1


@pytest.mark.parametrize("failure", ["corrupt", "disconnect"])
def test_retained_storage_fault_never_regenerates_baseline(retained, failure):
    engine, key, executor, control, _, _ = retained
    if failure == "corrupt":
        reference = coordinator.retained(engine, key, executor)["baseline_artifact"]
        executor.artifacts.data[reference.sha256] = b"corrupt"
        expected = BaselineUnavailable
    else:

        def disconnected(*args):
            raise ArtifactError("disconnected")

        executor.artifacts.get = disconnected
        expected = BaselinePending
    with pytest.raises(expected):
        coordinator.retained(engine, key, executor)
    assert control["calls"] == 1


def test_changed_retained_spec_rejected_before_artifact_read(retained):
    engine, key, executor, control, _, files = retained
    changed = dict(files)
    changed["spec/reference.py"] += b"\n# changed\n"
    executor.snapshots = SimpleNamespace(read=lambda _: SimpleNamespace(files=changed))
    with pytest.raises(ExecutionFailure, match="retained_spec_mismatch"):
        coordinator.retained(engine, key, executor)
    assert control["calls"] == 1
