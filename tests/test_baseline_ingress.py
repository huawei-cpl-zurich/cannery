import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from test_build_executor import Memory
from test_recipe_spec import recipe_spec

from cannery.artifacts import ArtifactError
from cannery.database import Base
from cannery.git_intake import GitSource
from cannery.git_intake.snapshots import SelectedTree, SnapshotStore
from cannery.numerics.baseline_format import baseline_key, pack_baseline
from cannery.numerics.baseline_state import BaselineState, claim_baseline, ensure_baseline
from cannery.numerics.baselines import BaselinePending, BaselineUnavailable, advance_baseline
from cannery.numerics.oracle_job import OracleFailure, OraclePending
from cannery.numerics.recipes import generate, parse_recipes
from cannery.registry.freeze import kernel_key
from cannery.registry.manifest import validate_spec


@pytest.fixture
def ingress(tmp_path, monkeypatch):
    files, _, _ = recipe_spec()
    spec = validate_spec(files, "spec")
    suite = parse_recipes(files["spec/recipes.yaml"])
    engine = create_engine("sqlite:///" + str(tmp_path / "db"))
    Base.metadata.create_all(engine)
    snapshots = SnapshotStore(tmp_path / "snapshots")
    binding = snapshots.commit(
        kernel_key("team", "identity"),
        GitSource(url="https://example.invalid/spec.git", commit="a" * 40, path="spec"),
        SelectedTree(files, dict.fromkeys(files, "100644"), "b" * 40),
    )
    key = baseline_key(spec.content_sha256, suite, spec.environment)
    ensure_baseline(engine, key, spec.content_sha256, binding)
    memory = Memory()
    control = {"calls": 0, "mode": None, "images": 0, "cleanups": 0}
    package = pack_baseline(
        spec.content_sha256,
        suite,
        spec.environment,
        "sha256:" + "a" * 64,
        {c.id: generate(c.inputs["x"], c.id, "x") for c in suite.cases},
    )
    reference = memory.put("outputs.tar", package)

    def image(*args):
        control["images"] += 1
        return 0, ("sha256:" + "a" * 64).encode()

    class Job:
        def __init__(self, *args):
            self.fence = args[-1]

        def progress(self, *args):
            self.fence()
            control["calls"] += 1
            if control["mode"] == "pending":
                raise OraclePending("lost Docker reply")
            if control["mode"] == "failed":
                raise OracleFailure("reference failed")
            return reference

        def cleanup(self):
            control["cleanups"] += 1

    monkeypatch.setattr("cannery.numerics.baselines.OracleJob", Job)
    monkeypatch.setattr("cannery.numerics.baselines.docker", image)

    def advance(fence=lambda: None):
        return advance_baseline(
            engine,
            key,
            spec,
            suite,
            snapshots=snapshots,
            artifacts=memory,
            root=tmp_path / "control",
            volume="test-control",
            image="cpu-reference",
            fence=fence,
        )

    yield advance, engine, key, control, memory, reference
    engine.dispose()


def test_ingress_reference_runs_once_and_future_registrations_reuse_ready_cache(ingress):
    advance, _, _, control, _, _ = ingress
    first = advance()
    second = advance()
    assert first == second and control["calls"] == control["images"] == 1
    assert first.assurance == "cached_cpu_reference" and first.cases == 3


def test_other_ingress_does_not_run_a_claimed_reference(ingress):
    advance, engine, key, control, _, _ = ingress
    assert claim_baseline(engine, key)
    with pytest.raises(BaselinePending, match="busy"):
        advance()
    assert control["calls"] == control["images"] == 0


def test_pending_operation_keeps_same_baseline_attempt(ingress):
    from datetime import UTC, datetime, timedelta

    advance, engine, key, control, _, _ = ingress
    control["mode"] = "pending"
    with pytest.raises(BaselinePending):
        advance()
    with Session(engine) as db, db.begin():
        row = db.get(BaselineState, key)
        attempt = row.attempt
        row.due_at = row.lease_until = datetime.now(UTC) - timedelta(seconds=1)
    control["mode"] = None
    assert advance().artifact
    with Session(engine) as db:
        row = db.get(BaselineState, key)
        assert row.attempt == attempt and row.attempts == 1


def test_missing_or_corrupt_cached_outputs_never_regenerate_reference(ingress):
    advance, engine, key, control, memory, reference = ingress
    advance()
    memory.data[reference.sha256] = b"corrupt"
    with pytest.raises(BaselineUnavailable, match="cache_invalid"):
        advance()
    assert control["calls"] == 1
    with Session(engine) as db:
        assert db.get(BaselineState, key).status == "ready"


def test_storage_disconnect_blocks_cached_read_without_reference_execution(ingress):
    advance, _, _, control, memory, _ = ingress
    advance()

    def offline(*args):
        raise ArtifactError("offline")

    memory.get = offline
    with pytest.raises(BaselinePending, match="storage_unavailable"):
        advance()
    assert control["calls"] == 1


def test_reference_failure_is_terminal_without_reexecution(ingress):
    advance, engine, key, control, _, _ = ingress
    control["mode"] = "failed"
    for _ in range(2):
        with pytest.raises(BaselineUnavailable, match="reference_failed"):
            advance()
    assert control["calls"] == 1 and control["cleanups"] == 1
    with Session(engine) as db:
        assert db.get(BaselineState, key).status == "failed"


def test_registration_ownership_loss_prevents_new_reference_work(ingress):
    advance, _, _, control, _, _ = ingress

    def lost():
        raise BaselinePending("registration lease lost")

    with pytest.raises(BaselinePending, match="lease lost"):
        advance(lost)
    assert control["calls"] == control["images"] == 0
