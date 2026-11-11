from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session
from test_build_state import candidate
from test_submission_requests import records as records

from cannery.numerics.state import (
    NumericalResult,
    NumericalState,
    claim_numerical,
    current_numerical,
    due_numericals,
    finish_numerical,
    heartbeat,
    postpone_numerical,
)


def queue(records):
    engine, _ = records
    key = candidate(records)
    with Session(engine) as db, db.begin():
        db.add(NumericalState(id=key))
    return key


def test_reclaim_retains_remote_attempt_and_fences_old_owner(records):
    engine, _ = records
    key = queue(records)
    first = claim_numerical(engine, key)
    assert heartbeat(engine, first) and claim_numerical(engine, key) is None
    assert postpone_numerical(engine, first)
    assert not heartbeat(engine, first) and claim_numerical(engine, key) is None
    with Session(engine) as db, db.begin():
        db.get(NumericalState, key).lease_until = datetime.now(UTC) - timedelta(seconds=1)
    second = claim_numerical(engine, key)
    assert second.attempt == first.attempt and second.token != first.token
    assert not finish_numerical(engine, first, report={"passed": True})
    with Session(engine) as db:
        assert current_numerical(db, second) and db.get(NumericalState, key).attempts == 1


def test_bounded_retry_has_distinct_immutable_results(records):
    engine, _ = records
    key = queue(records)
    claims = []
    for _ in range(3):
        with Session(engine) as db, db.begin():
            db.get(NumericalState, key).due_at = datetime.now(UTC)
        claim = claim_numerical(engine, key)
        claims.append(claim)
        assert finish_numerical(engine, claim, error="device_unavailable", retryable=True)
        assert claim_numerical(engine, key) is None
    assert len({c.attempt for c in claims}) == 3
    with Session(engine) as db:
        assert db.get(NumericalState, key).status == "failed"
        results = list(db.scalars(select(NumericalResult)))
        assert len(results) == 3
        results[0].error = "overwrite"
        with pytest.raises(ValueError, match="immutable"):
            db.commit()


@pytest.mark.parametrize("passed", [False, True])
def test_report_is_terminal_and_cannot_be_completed_twice(records, passed):
    engine, _ = records
    key = queue(records)
    assert str(key) in due_numericals(engine)
    claim = claim_numerical(engine, key)
    assert finish_numerical(engine, claim, report={"passed": passed, "cases": []})
    assert not finish_numerical(engine, claim, report={"passed": True})
    assert claim_numerical(engine, key) is None
    with Session(engine) as db:
        assert db.get(NumericalState, key).status == ("passed" if passed else "failed")
        assert db.get(NumericalResult, claim.attempt).report["passed"] is passed


def test_expired_attempt_gets_one_terminal_result(records):
    engine, _ = records
    key = queue(records)
    claim = claim_numerical(engine, key)
    with Session(engine) as db, db.begin():
        row = db.get(NumericalState, key)
        row.deadline = row.lease_until = datetime.now(UTC) - timedelta(seconds=1)
    assert claim_numerical(engine, key) is None
    assert claim_numerical(engine, key) is None
    with Session(engine) as db:
        assert db.get(NumericalResult, claim.attempt).error == "deadline_exceeded"
        assert len(list(db.scalars(select(NumericalResult)))) == 1
