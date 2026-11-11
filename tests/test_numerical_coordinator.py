import json
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest
from sqlalchemy.orm import Session
from test_build_coordinator import Executor
from test_build_state import candidate
from test_submission_requests import records as records

from cannery.artifacts import ArtifactRef
from cannery.builds.executor import ExecutionFailure
from cannery.builds.remote import RemotePending
from cannery.builds.state import claim_build, finish_build
from cannery.numerics import coordinator
from cannery.numerics.baselines import BaselineUnavailable
from cannery.numerics.state import (
    NumericalResult,
    NumericalState,
    claim_numerical,
    finish_numerical,
)


def test_legacy_binary_is_explicitly_blocked_and_discovery_idempotent(records):
    engine, _ = records
    key = candidate(records)
    claim = claim_build(engine, key)
    ref = ArtifactRef(repository="private", revision="a" * 64, sha256="a" * 64, path="binary.tar")
    assert finish_build(engine, claim, artifact=ref)
    coordinator.discover(engine)
    coordinator.discover(engine)
    with Session(engine) as db:
        row = db.get(NumericalState, key)
        assert row.status == "blocked" and row.error == "recipe_baseline_required"
    assert claim_numerical(engine, key) is None


@pytest.fixture
def operation(records, monkeypatch, tmp_path):
    engine, _ = records
    key = candidate(records)
    with Session(engine) as db, db.begin():
        db.add(NumericalState(id=key))
    executor = Executor(tmp_path / "numerics")
    executor.validate = MagicMock(return_value={"passed": True, "cases": []})
    monkeypatch.setattr(coordinator, "retained", lambda *args: {})
    return engine, key, executor


def test_numerical_result_commits_after_cleanup_once(operation):
    engine, key, executor = operation
    coordinator.process(engine, key, lambda _: executor)
    assert executor.cleaned
    with Session(engine) as db:
        row = db.get(NumericalState, key)
        assert row.status == "passed"
        assert db.get(NumericalResult, row.attempt_id).report["passed"]
    coordinator.process(engine, key, lambda _: executor)
    executor.validate.assert_called_once()


@pytest.mark.parametrize("failure", [RemotePending("offline"), RemotePending("busy")])
def test_lost_reply_and_busy_device_resume_same_attempt(operation, failure):
    engine, key, executor = operation
    executor.validate.side_effect = failure
    coordinator.process(engine, key, lambda _: executor)
    with Session(engine) as db, db.begin():
        row = db.get(NumericalState, key)
        original = row.attempt_id
        assert row.attempts == 1 and row.status == "validating"
        row.lease_until = datetime.now(UTC) - timedelta(seconds=1)
    executor.validate.side_effect = None
    coordinator.process(engine, key, lambda _: executor)
    with Session(engine) as db:
        row = db.get(NumericalState, key)
        assert row.attempt_id == original and row.status == "passed"


def test_corrupt_cache_blocks_without_npu_dispatch(operation, monkeypatch):
    engine, key, executor = operation

    def corrupt(*args):
        raise BaselineUnavailable("baseline_cache_invalid")

    monkeypatch.setattr(coordinator, "retained", corrupt)
    coordinator.process(engine, key, lambda _: executor)
    executor.validate.assert_not_called()
    with Session(engine) as db:
        assert db.get(NumericalState, key).status == "blocked"


def test_terminal_cleanup_retains_uncertain_remote_ownership(operation, tmp_path):
    engine, key, executor = operation
    claim = claim_numerical(engine, key)
    finish_numerical(engine, claim, error="deadline_exceeded")
    journal = tmp_path / claim.attempt.hex
    journal.mkdir()
    executor.remote.journal = journal
    (journal / "runtime-0.dispatch").write_text("uncertain")
    executor.cleanup = MagicMock(side_effect=RemotePending("uncertain"))
    coordinator.cleanup_finished(engine, tmp_path, lambda _: executor)
    assert not (journal / "cleanup-complete").exists()
    executor.cleanup.side_effect = None
    coordinator.cleanup_finished(engine, tmp_path, lambda _: executor)
    assert (journal / "cleanup-complete").exists()


def test_ownership_loss_during_retained_reads_fences_error_cleanup(operation, monkeypatch):
    engine, key, executor = operation
    (executor.remote.journal / "runtime-0.dispatch").write_text("submitted")
    dispatched = []

    def cleanup(attempt):
        executor.fence()
        dispatched.append(attempt)

    executor.cleanup = cleanup

    def interrupted(*args):
        with Session(engine) as db, db.begin():
            db.get(NumericalState, key).lease_until = datetime.now(UTC) - timedelta(seconds=1)
        assert claim_numerical(engine, key) is not None
        raise BaselineUnavailable("baseline_cache_invalid")

    monkeypatch.setattr(coordinator, "retained", interrupted)
    coordinator.process(engine, key, lambda _: executor)
    assert dispatched == []
    with Session(engine) as db:
        row = db.get(NumericalState, key)
        assert row.status == "validating" and row.attempts == 1
        assert db.get(NumericalResult, row.attempt_id) is None


def receipt_with_bad_output(executor, raw):
    journal = executor.remote.journal
    (journal / "runtime-0.dispatch").write_text("submitted")
    (journal / "runtime-0.json").write_text(json.dumps({"handle": "remote:test:job:runtime-0"}))
    (journal / "results.json").write_text(raw)
    calls = []

    def call(*args):
        calls.append(args)
        return {"state": "completed", "content": "bounded fixture log"}

    executor.remote.call = call
    return calls


@pytest.mark.parametrize(
    "raw", ["{", "null", "42", "[]", '{"handle":"remote:foreign:job:private"}']
)
def test_invalid_output_is_not_parsed_as_a_receipt_and_fails_terminally(operation, raw):
    engine, key, executor = operation
    calls = receipt_with_bad_output(executor, raw)
    executor.validate.side_effect = ExecutionFailure("invalid_numerical_output")
    coordinator.process(engine, key, lambda _: executor)
    with Session(engine) as db:
        row = db.get(NumericalState, key)
        assert (
            row.status == "failed" and row.error == "invalid_numerical_output" and row.attempts == 1
        )
    assert len(calls) == 2 and all(c[1] == "remote:test:job:runtime-0" for c in calls)
    assert (executor.remote.journal / "cleanup-complete").exists()


@pytest.mark.parametrize("raw", ["null", '{"handle":"remote:foreign:job:private"}'])
def test_terminal_cleanup_ignores_untrusted_result_json(operation, tmp_path, raw):
    engine, key, executor = operation
    claim = claim_numerical(engine, key)
    assert finish_numerical(engine, claim, error="deadline_exceeded")
    executor.remote.journal = tmp_path / claim.attempt.hex
    executor.remote.journal.mkdir()
    calls = receipt_with_bad_output(executor, raw)
    coordinator.cleanup_finished(engine, tmp_path, lambda _: executor)
    assert executor.cleaned and (executor.remote.journal / "cleanup-complete").exists()
    assert all(c[1] == "remote:test:job:runtime-0" for c in calls)
