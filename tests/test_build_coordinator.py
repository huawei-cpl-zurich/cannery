from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from sqlalchemy import delete, select
from sqlalchemy.orm import Session
from test_build_executor import Memory
from test_build_sandbox import profile
from test_build_state import candidate
from test_submission_requests import records as records

from cannery.artifacts import ArtifactRef
from cannery.builds.coordinator import discover, process
from cannery.builds.executor import ExecutionFailure
from cannery.builds.remote import RemotePending
from cannery.builds.state import BuildAttempt, BuildResult, BuildState


class Executor:
    def __init__(self, root, failure=None):
        self.profile, self.wheel_sha256 = profile(root / "remote"), "a" * 64
        root.mkdir(exist_ok=True)
        self.remote = SimpleNamespace(journal=root)
        self.artifacts, self.failure = Memory(), failure
        self.cleaned = False

    def execute(self, candidate, attempt, fence):
        fence()
        if self.failure:
            raise self.failure
        return ArtifactRef(
            repository="private", revision="a" * 64, sha256="a" * 64, path="package.tar"
        )

    def cleanup(self, attempt):
        self.cleaned = True


def test_source_discovery_ignores_legacy_backlog_and_is_idempotent(records):
    engine, _ = records
    for _ in range(55):
        candidate(records, native=False)
    key = candidate(records)
    with engine.begin() as connection:
        connection.execute(delete(BuildState).where(BuildState.id == key))
    discover(engine)
    discover(engine)
    with Session(engine) as session:
        assert list(session.scalars(select(BuildState.id))) == [key]


def test_private_admission_and_evidence_are_persisted_only_after_cleanup(records, tmp_path):
    engine, _ = records
    key = candidate(records)
    executor = Executor(tmp_path / "control")
    process(engine, key, lambda claim: executor)
    assert executor.cleaned
    with Session(engine) as session:
        state = session.get(BuildState, key)
        result = session.get(BuildResult, state.attempt_id)
        assert state.status == "awaiting_validation" and result.status == "complete"
        assert result.evidence["loader_sha256"] == "a" * 64
        assert result.evidence["worker"]["target"] == "test-a5-host"


def test_pending_transport_reuses_attempt_after_coordinator_restart(records, tmp_path):
    engine, _ = records
    key = candidate(records)
    first = Executor(tmp_path / "control", RemotePending("network outage"))
    process(engine, key, lambda claim: first)
    assert not first.cleaned
    with Session(engine) as session, session.begin():
        state = session.get(BuildState, key)
        original = state.attempt_id
        state.lease_until = datetime.now(UTC) - timedelta(seconds=1)
    second = Executor(tmp_path / "control")
    process(engine, key, lambda claim: second)
    with Session(engine) as session:
        assert session.get(BuildState, key).attempt_id == original
        assert len(session.scalars(select(BuildAttempt)).all()) == 1
        assert session.get(BuildState, key).status == "awaiting_validation"


def test_permanent_build_failure_remains_failed_with_private_evidence(records, tmp_path):
    engine, _ = records
    key = candidate(records)
    executor = Executor(tmp_path / "control", ExecutionFailure("compile_failed"))
    process(engine, key, lambda claim: executor)
    with Session(engine) as session:
        state = session.get(BuildState, key)
        result = session.get(BuildResult, state.attempt_id)
        assert state.status == "failed" and state.error == "compile_failed"
        assert result.evidence["worker"]["image"].startswith("sha256:")


def test_cleanup_outage_defers_admission_and_resumes_same_attempt(records, tmp_path):
    engine, _ = records
    key = candidate(records)
    executor = Executor(tmp_path / "control")

    def unavailable(attempt):
        raise RemotePending("device ownership cleanup uncertain")

    executor.cleanup = unavailable
    process(engine, key, lambda claim: executor)
    with Session(engine) as session, session.begin():
        state = session.get(BuildState, key)
        original = state.attempt_id
        assert state.status == "building"
        assert session.get(BuildResult, original) is None
        state.lease_until = datetime.now(UTC) - timedelta(seconds=1)
    assert not (executor.remote.journal / "cleanup-complete").exists()
    recovered = Executor(tmp_path / "control")
    process(engine, key, lambda claim: recovered)
    with Session(engine) as session:
        assert session.get(BuildState, key).attempt_id == original
        assert session.get(BuildState, key).status == "awaiting_validation"
    assert recovered.cleaned


def test_orphan_cleanup_retains_uncertain_device_and_retries(records, tmp_path):
    from cannery.builds.coordinator import cleanup_finished
    from cannery.builds.state import claim_build, finish_build

    engine, _ = records
    key = candidate(records)
    claim = claim_build(engine, key)
    assert finish_build(engine, claim, error="deadline_exceeded")
    journal = tmp_path / claim.attempt.hex
    executor = Executor(journal)
    (journal / "compile.dispatch").write_text("dispatched")

    def unavailable(attempt):
        raise RemotePending("removal uncertain")

    executor.cleanup = unavailable
    cleanup_finished(engine, tmp_path, lambda claim: executor)
    assert not (journal / "cleanup-complete").exists()
    recovered = Executor(journal)
    cleanup_finished(engine, tmp_path, lambda claim: recovered)
    assert recovered.cleaned and (journal / "cleanup-complete").exists()
