import json
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session
from test_build_coordinator import Executor
from test_submission_requests import records as records

from cannery.artifacts import ArtifactError
from cannery.benchmark import coordinator
from cannery.benchmark.state import PerformanceJob, PerformanceResult, ensure_job
from cannery.builds.executor import ExecutionFailure
from cannery.builds.remote import RemotePending

KEY = "a" * 64


@pytest.fixture
def operation(records, tmp_path, monkeypatch):
    engine, _ = records
    ensure_job(engine, KEY, "reference", {})
    executor = Executor(tmp_path / "benchmark")
    artifact = executor.artifacts.put("samples.json", b"raw timing")
    executor.benchmark = MagicMock(return_value=({"passed": True, "cases": []}, artifact))
    (executor.remote.journal / "runtime-0.dispatch").write_text("owned")
    (executor.remote.journal / "runtime-0.json").write_text(
        json.dumps({"handle": "remote:test:job:one"})
    )
    executor.remote.call = MagicMock(return_value={"state": "completed"})
    monkeypatch.setattr(coordinator, "retained", MagicMock(return_value={}))
    monkeypatch.setattr(coordinator, "evidence", MagicMock(return_value={"operations": {}}))
    return engine, executor


def expire(engine):
    with Session(engine) as db, db.begin():
        row = db.get(PerformanceJob, KEY)
        row.due_at = row.lease_until = datetime.now(UTC) - timedelta(seconds=1)


def test_success_finalizes_once_after_cleanup(operation):
    engine, executor = operation
    coordinator.process(engine, KEY, lambda _: executor)
    assert executor.cleaned
    with Session(engine) as db:
        row = db.get(PerformanceJob, KEY)
        assert row.status == "ready"
        assert db.get(PerformanceResult, row.attempt).report["passed"]
    assert (executor.remote.journal / "cleanup-complete").exists()
    coordinator.process(engine, KEY, lambda _: executor)
    executor.benchmark.assert_called_once()


def test_transport_loss_resumes_same_attempt(operation):
    engine, executor = operation
    executor.benchmark.side_effect = RemotePending("lost reply")
    coordinator.process(engine, KEY, lambda _: executor)
    with Session(engine) as db:
        original = db.get(PerformanceJob, KEY).attempt
    assert not executor.cleaned
    executor.benchmark.side_effect = None
    expire(engine)
    coordinator.process(engine, KEY, lambda _: executor)
    with Session(engine) as db:
        row = db.get(PerformanceJob, KEY)
        assert row.attempt == original and row.attempts == 1 and row.status == "ready"


def test_failure_checkpoint_survives_cleanup_then_artifact_outage(operation, monkeypatch):
    engine, executor = operation
    executor.benchmark.side_effect = ExecutionFailure("device_failed", retryable=True)
    coordinator.evidence.side_effect = ArtifactError("offline")
    coordinator.process(engine, KEY, lambda _: executor)
    assert executor.cleaned and (executor.remote.journal / "benchmark-failure.json").exists()
    monkeypatch.setattr(
        coordinator,
        "retained",
        MagicMock(side_effect=AssertionError("must not reread cleaned inputs")),
    )
    coordinator.evidence.side_effect = None
    expire(engine)
    coordinator.process(engine, KEY, lambda _: executor)
    executor.benchmark.assert_called_once()
    with Session(engine) as db:
        row = db.get(PerformanceJob, KEY)
        assert row.status == "queued" and row.attempts == 1
        assert db.get(PerformanceResult, row.attempt).error == "device_failed"


def test_lost_owner_never_cleans_or_writes_failure(operation, monkeypatch):
    engine, executor = operation

    def lost(*_):
        with Session(engine) as db, db.begin():
            db.get(PerformanceJob, KEY).lease = uuid4()
        raise ExecutionFailure("failure_after_lease_loss")

    monkeypatch.setattr(coordinator, "retained", lost)
    coordinator.process(engine, KEY, lambda _: executor)
    assert not executor.cleaned
    assert not (executor.remote.journal / "benchmark-failure.json").exists()


def test_terminal_cleanup_ignores_untrusted_json_and_restores_profile(operation):
    engine, executor = operation
    coordinator.process(engine, KEY, lambda _: executor)
    journal = executor.remote.journal
    with Session(engine) as db:
        attempt = db.get(PerformanceJob, KEY).attempt
    moved = journal.with_name(attempt.hex)
    journal.rename(moved)
    journal = executor.remote.journal = moved
    (journal / "cleanup-complete").unlink()
    (journal / "results.json").write_text("null")
    (journal / "context.json").write_text(
        json.dumps({"profile": executor.profile.model_dump(mode="json")})
    )
    executor.cleaned = False
    coordinator.cleanup_finished(engine, journal.parent, lambda _: executor)
    assert executor.cleaned and (journal / "cleanup-complete").exists()
    assert not (journal / "results.json").exists()
    assert all("one" in str(call) for call in executor.remote.call.call_args_list)
