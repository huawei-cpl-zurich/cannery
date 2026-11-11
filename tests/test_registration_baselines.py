from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.orm import Session
from test_registration_state import job as job

from cannery.git_intake import SnapshotBinding
from cannery.numerics.baseline_state import ensure_baseline
from cannery.registry.registrations import (
    Registration,
    claim_registration,
    postpone_registration,
    retain_baseline_registration,
)


def prepare(engine, key):
    claim = claim_registration(engine, key)
    binding = SnapshotBinding(
        source=claim.source,
        internal_repository="team",
        internal_snapshot_sha="a" * 40,
        external_tree_id="b" * 40,
        content_sha256="c" * 64,
    )
    ensure_baseline(engine, "1" * 64, "2" * 64, binding)
    assert retain_baseline_registration(engine, claim, binding, "1" * 64)
    return claim, binding


def ready(engine, key):
    with Session(engine) as db, db.begin():
        row = db.get(Registration, key)
        row.due_at = row.lease_until = datetime.now(UTC) - timedelta(seconds=1)


def test_baseline_polling_preserves_snapshot_and_intake_attempt_budget(job):
    engine, key = job
    first, binding = prepare(engine, key)
    assert postpone_registration(engine, first)
    assert claim_registration(engine, key) is None
    for _ in range(5):
        ready(engine, key)
        claim = claim_registration(engine, key)
        assert claim.prepared_snapshot == binding and claim.baseline_key == "1" * 64
        assert postpone_registration(engine, claim)
    with Session(engine) as db:
        assert db.get(Registration, key).attempts == 1


def test_prepared_registration_still_has_deadline_and_owner_fencing(job):
    engine, key = job
    first, binding = prepare(engine, key)
    assert postpone_registration(engine, first)
    ready(engine, key)
    second = claim_registration(engine, key)
    assert second.token != first.token
    assert not retain_baseline_registration(engine, first, binding, "1" * 64)
    assert not postpone_registration(engine, first)
    with Session(engine) as db, db.begin():
        db.get(Registration, key).deadline = datetime.now(UTC) - timedelta(seconds=1)
    ready(engine, key)
    assert claim_registration(engine, key) is None
    with Session(engine) as db:
        assert db.get(Registration, key).error == "deadline_exceeded"


def test_unprepared_legacy_registration_cannot_skip_intake_retries(job):
    engine, key = job
    claim = claim_registration(engine, key)
    with pytest.raises(ValueError, match="Only prepared"):
        postpone_registration(engine, claim)


def test_prepared_snapshot_and_baseline_identity_cannot_be_replaced(job):
    engine, key = job
    claim, binding = prepare(engine, key)
    ensure_baseline(engine, "3" * 64, "2" * 64, binding)
    with pytest.raises(ValueError, match="cannot change"):
        retain_baseline_registration(engine, claim, binding, "3" * 64)
    with pytest.raises(ValueError):
        retain_baseline_registration(
            engine,
            claim,
            binding.model_copy(
                update={"source": binding.source.model_copy(update={"commit": "f" * 40})}
            ),
            "1" * 64,
        )
