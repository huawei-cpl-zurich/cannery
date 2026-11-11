from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from cannery.artifacts import ArtifactRef
from cannery.database import Base
from cannery.git_intake import GitSource, SnapshotBinding
from cannery.numerics.baseline_state import (
    BaselineState,
    claim_baseline,
    current_baseline,
    ensure_baseline,
    finish_baseline,
    postpone_baseline,
)


@pytest.fixture
def cache(tmp_path):
    engine = create_engine("sqlite:///" + str(tmp_path / "db.sqlite"))
    Base.metadata.create_all(engine)
    source = GitSource(url="https://example.invalid/spec.git", commit="a" * 40, path="spec")
    snapshot = SnapshotBinding(
        source=source,
        internal_repository="team",
        internal_snapshot_sha="b" * 40,
        external_tree_id="c" * 40,
        content_sha256="d" * 64,
    )
    ensure_baseline(engine, "1" * 64, "2" * 64, snapshot)
    yield engine, snapshot
    engine.dispose()


def expire(engine):
    with Session(engine) as db, db.begin():
        db.get(BaselineState, "1" * 64).lease_until = datetime.now(UTC) - timedelta(seconds=1)


def reference():
    return ArtifactRef(
        repository="private", revision="e" * 64, sha256="e" * 64, path="baseline.tar"
    )


def test_duplicate_ingress_reuses_one_cache_and_retains_first_binding(cache):
    engine, snapshot = cache
    ensure_baseline(engine, "1" * 64, "2" * 64, snapshot)
    with Session(engine) as db:
        assert len(db.scalars(select(BaselineState)).all()) == 1
        assert db.get(BaselineState, "1" * 64).snapshot == snapshot.model_dump(mode="json")
    with pytest.raises(ValueError):
        ensure_baseline(engine, "1" * 64, "3" * 64, snapshot)


def test_restart_keeps_attempt_but_fences_old_owner(cache):
    engine, _ = cache
    first = claim_baseline(engine, "1" * 64)
    assert first and claim_baseline(engine, "1" * 64) is None
    expire(engine)
    second = claim_baseline(engine, "1" * 64)
    assert second.attempt == first.attempt and second.token != first.token
    assert not finish_baseline(engine, first, artifact=reference())
    with Session(engine) as db:
        assert current_baseline(db, second) is not None
    assert finish_baseline(engine, second, artifact=reference())
    assert claim_baseline(engine, "1" * 64) is None


def test_pending_operation_defers_without_incrementing_attempt(cache):
    engine, _ = cache
    first = claim_baseline(engine, "1" * 64)
    assert postpone_baseline(engine, first)
    assert claim_baseline(engine, "1" * 64) is None
    with Session(engine) as db, db.begin():
        row = db.get(BaselineState, "1" * 64)
        row.due_at = row.lease_until = datetime.now(UTC) - timedelta(seconds=1)
    second = claim_baseline(engine, "1" * 64)
    assert first.attempt == second.attempt
    with Session(engine) as db:
        assert db.get(BaselineState, "1" * 64).attempts == 1
    assert not postpone_baseline(engine, first)


def test_completed_reference_is_immutable_and_reused_on_new_registration(cache):
    engine, snapshot = cache
    claim = claim_baseline(engine, "1" * 64)
    assert finish_baseline(engine, claim, artifact=reference())
    ensure_baseline(engine, "1" * 64, "2" * 64, snapshot)
    assert claim_baseline(engine, "1" * 64) is None
    with Session(engine) as db:
        row = db.get(BaselineState, "1" * 64)
        assert row.artifact == reference().model_dump(mode="json")
        row.error = "changed"
        with pytest.raises(ValueError, match="immutable"):
            db.commit()
    with Session(engine) as db:
        db.delete(db.get(BaselineState, "1" * 64))
        with pytest.raises(ValueError, match="immutable"):
            db.commit()


def test_known_failed_attempts_retry_with_backoff_then_stop(cache):
    engine, _ = cache
    attempts = []
    for _ in range(3):
        claim = claim_baseline(engine, "1" * 64)
        attempts.append(claim.attempt)
        assert finish_baseline(engine, claim, error="worker_unavailable", retryable=True)
        assert claim_baseline(engine, "1" * 64) is None
        with Session(engine) as db, db.begin():
            db.get(BaselineState, "1" * 64).due_at = datetime.now(UTC) - timedelta(seconds=1)
    assert len(set(attempts)) == 3
    with Session(engine) as db:
        assert db.get(BaselineState, "1" * 64).status == "failed"


def test_deadline_and_reference_error_are_terminal(cache):
    engine, _ = cache
    claim = claim_baseline(engine, "1" * 64)
    with Session(engine) as db, db.begin():
        row = db.get(BaselineState, "1" * 64)
        row.deadline = row.lease_until = datetime.now(UTC) - timedelta(seconds=1)
    assert not finish_baseline(engine, claim, artifact=reference())
    assert claim_baseline(engine, "1" * 64) is None
    with Session(engine) as db:
        assert db.get(BaselineState, "1" * 64).error == "deadline_exceeded"


def test_only_pinned_private_success_and_exclusive_outcomes_allowed(cache):
    engine, _ = cache
    claim = claim_baseline(engine, "1" * 64)
    with pytest.raises(ValueError):
        finish_baseline(engine, claim)
    with pytest.raises(ValueError):
        finish_baseline(engine, claim, artifact=reference(), error="bad")
    with pytest.raises(ValueError):
        finish_baseline(
            engine, claim, artifact=reference().model_copy(update={"repository": "external"})
        )
    assert finish_baseline(engine, claim, error="reference_failed")
    assert claim_baseline(engine, "1" * 64) is None
