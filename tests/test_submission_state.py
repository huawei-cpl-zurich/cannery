from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session
from test_submission_requests import initial
from test_submission_requests import records as records

from cannery.submissions.state import (
    IntakeState,
    claim_intake,
    clear_scratch,
    current_intake,
    due_intakes,
    fail_intake,
    pending_scratch,
    prepare_scratch,
)


def test_duplicate_delivery_and_expired_lease_are_fenced(records):
    engine, _ = records
    key = initial(records)
    first = claim_intake(engine, key)
    assert first is not None
    assert claim_intake(engine, key) is None
    token = prepare_scratch(engine, first)
    with Session(engine) as session, session.begin():
        session.get(IntakeState, key).lease_until = datetime.now(UTC) - timedelta(seconds=1)
    second = claim_intake(engine, key)
    assert second is not None and second.previous_scratch == token and second.token != first.token
    with pytest.raises(ValueError):
        prepare_scratch(engine, second)
    fail_intake(engine, first, "stale", False)
    with Session(engine) as session:
        assert current_intake(session, second) is not None
    clear_scratch(engine, key, token)
    new = prepare_scratch(engine, second)
    clear_scratch(engine, key, token)
    with Session(engine) as session:
        assert session.get(IntakeState, key).scratch_token == new


def test_retry_budget_and_terminal_cleanup_survive(records):
    engine, _ = records
    key = initial(records)
    for _ in range(3):
        with Session(engine) as session, session.begin():
            session.get(IntakeState, key).due_at = datetime.now(UTC)
        claim = claim_intake(engine, key)
        if claim.previous_scratch:
            clear_scratch(engine, key, claim.previous_scratch)
        token = prepare_scratch(engine, claim)
        fail_intake(engine, claim, "intake_unavailable", True)
    assert claim_intake(engine, key) is None
    assert pending_scratch(engine, key) == token
    assert str(key) in due_intakes(engine)
    clear_scratch(engine, key, token)
    assert pending_scratch(engine, key) is None
    with Session(engine) as session:
        state = session.get(IntakeState, key)
        assert state.status == "failed" and state.attempts == 3


def test_expired_deadline_and_permanent_failure_do_not_retry(records):
    engine, _ = records
    expired = initial(records)
    with Session(engine) as session, session.begin():
        session.get(IntakeState, expired).deadline = datetime.now(UTC) - timedelta(seconds=1)
    assert claim_intake(engine, expired) is None
    other = initial(records)
    claim = claim_intake(engine, other)
    fail_intake(engine, claim, "invalid_submission", False)
    assert claim_intake(engine, other) is None
    with Session(engine) as session:
        assert session.get(IntakeState, expired).error == "deadline_exceeded"
        assert session.get(IntakeState, other).attempts == 1


def test_cleanup_backlog_cannot_starve_active_work(records):
    engine, _ = records
    cleanup = []
    for _ in range(120):
        key = initial(records)
        cleanup.append(str(key))
        with Session(engine) as session, session.begin():
            job = session.get(IntakeState, key)
            job.status, job.scratch_token = "failed", uuid4()
    active = initial(records)
    seen = set()
    for _ in range(3):
        dispatched = due_intakes(engine)
        assert str(active) in dispatched
        seen.update(dispatched)
    assert set(cleanup) <= seen
