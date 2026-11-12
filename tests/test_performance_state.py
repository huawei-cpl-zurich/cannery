from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session
from test_build_state import candidate
from test_submission_requests import records as records

from cannery.artifacts import ArtifactRef
from cannery.benchmark.state import (
    PerformanceJob,
    PerformanceResult,
    claim_job,
    current_job,
    due_jobs,
    ensure_job,
    finish_job,
    heartbeat,
    postpone_job,
)

KEY = "a" * 64
ARTIFACT = ArtifactRef(
    repository="private", revision="b" * 64, path="timing/raw.json", sha256="b" * 64
)


def queue(records):
    engine, _ = records
    ensure_job(engine, KEY, "reference", {"suite": "one"})
    return engine


def due(engine):
    with Session(engine) as db, db.begin():
        row = db.get(PerformanceJob, KEY)
        row.due_at = datetime.now(UTC) - timedelta(seconds=1)
        row.lease_until = row.due_at


def test_deduplication_and_identity_conflict(records):
    engine = queue(records)
    ensure_job(engine, KEY, "reference", {"suite": "one"})
    with pytest.raises(ValueError, match="Conflicting"):
        ensure_job(engine, KEY, "reference", {"suite": "two"})
    key = candidate(records)
    ensure_job(
        engine, "c" * 64, "candidate", {"binary": "one"}, candidate_id=key, reference_key=KEY
    )
    with pytest.raises(ValueError):
        ensure_job(engine, "d" * 64, "candidate", {}, candidate_id=key, reference_key=KEY)


def test_restart_retains_attempt_but_fences_stale_owner(records):
    engine = queue(records)
    assert KEY in due_jobs(engine)
    first = claim_job(engine, KEY)
    assert heartbeat(engine, first) and claim_job(engine, KEY) is None
    assert postpone_job(engine, first) and not heartbeat(engine, first)
    assert claim_job(engine, KEY) is None
    due(engine)
    second = claim_job(engine, KEY)
    assert first.attempt == second.attempt and first.token != second.token
    assert not finish_job(engine, first, error="stale")
    with Session(engine) as db:
        assert current_job(db, second)
    assert finish_job(engine, second, report={"cases": []}, artifact=ARTIFACT)
    assert not finish_job(engine, second, error="duplicate")
    assert claim_job(engine, KEY) is None
    with Session(engine) as db:
        row = db.get(PerformanceJob, KEY)
        row.status = "queued"
        with pytest.raises(ValueError, match="immutable"):
            db.commit()


def test_retry_cap_and_immutable_results(records):
    engine = queue(records)
    attempts = []
    for _ in range(3):
        due(engine)
        claim = claim_job(engine, KEY)
        attempts.append(claim.attempt)
        assert finish_job(engine, claim, error="device_failed", retryable=True)
        assert claim_job(engine, KEY) is None
    assert len(set(attempts)) == 3
    with Session(engine) as db:
        assert db.get(PerformanceJob, KEY).status == "failed"
        results = list(db.scalars(select(PerformanceResult)))
        assert len(results) == 3
        results[0].error = "rewrite"
        with pytest.raises(ValueError, match="immutable"):
            db.commit()


def test_deadline_has_one_result_and_no_fresh_dispatch(records):
    engine = queue(records)
    claim = claim_job(engine, KEY)
    with Session(engine) as db, db.begin():
        row = db.get(PerformanceJob, KEY)
        row.deadline = row.lease_until = datetime.now(UTC) - timedelta(seconds=1)
    assert not heartbeat(engine, claim)
    assert claim_job(engine, KEY) is None and claim_job(engine, KEY) is None
    with Session(engine) as db:
        assert db.get(PerformanceResult, claim.attempt).error == "deadline_exceeded"
        assert len(list(db.scalars(select(PerformanceResult)))) == 1


def test_blocked_and_success_contract(records):
    engine = queue(records)
    claim = claim_job(engine, KEY)
    with pytest.raises(ValueError):
        finish_job(engine, claim, report={})
    assert finish_job(engine, claim, error="reference_unsupported", blocked=True)
    with Session(engine) as db:
        assert db.get(PerformanceJob, KEY).status == "blocked"
    assert claim_job(engine, KEY) is None
