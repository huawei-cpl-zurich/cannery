from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session
from test_submission_requests import initial
from test_submission_requests import records as records
from test_tensor_submission_reader import tensor_files

from cannery.builds.coordinator import discover, process
from cannery.builds.state import BuildState, claim_build, due_builds, queue_build
from cannery.submissions.manifest import validate_tensor_submission
from cannery.submissions.models import Candidate


@pytest.fixture(params=["ascendc", "catlass", "triton"])
def tensor(records, request):
    engine, _ = records
    key = initial(records)
    valid = validate_tensor_submission(tensor_files(request.param), "candidate")
    with Session(engine) as db, db.begin():
        db.add(
            Candidate(
                id=key,
                binding_id=key,
                binding={},
                manifest=valid.manifest.model_dump(mode="json"),
                snapshot={},
                content_sha256=valid.content_sha256,
                file_sha256=valid.file_sha256,
            )
        )
    return engine, key


def test_rollout_disabled_discovery_and_claim_leave_tensor_untouched(tensor):
    engine, key = tensor
    discover(engine)
    with Session(engine) as db, db.begin():
        candidate = db.get(Candidate, key)
        assert db.get(BuildState, key) is None
        assert queue_build(db, candidate) is False
        assert queue_build(db, candidate, tensor_enabled=True)
    assert due_builds(engine) == []
    assert claim_build(engine, key) is None
    assert due_builds(engine, tensor_enabled=True) == [str(key)]
    assert claim_build(engine, key, device="a3", tensor_enabled=True) is None
    with Session(engine) as db:
        assert db.get(BuildState, key).attempts == 0


def test_enabled_discovery_is_idempotent_and_claims_one_fenced_attempt(tensor):
    engine, key = tensor
    for _ in range(2):
        discover(engine, tensor_enabled=True)
    with Session(engine) as db:
        assert list(db.scalars(select(BuildState.id))) == [key]
    claim = claim_build(engine, key, tensor_enabled=True)
    assert claim is not None
    assert claim_build(engine, key, tensor_enabled=True) is None
    with Session(engine) as db, db.begin():
        row = db.get(BuildState, key)
        row.lease_until = datetime.now(UTC) - timedelta(seconds=1)
    resumed = claim_build(engine, key, tensor_enabled=True)
    assert resumed.attempt == claim.attempt and resumed.token != claim.token


def test_storage_disconnect_keeps_build_attempt_for_bounded_retry(tensor):
    engine, key = tensor
    discover(engine, tensor_enabled=True)

    def unavailable(claim):
        raise OSError("injected storage disconnect")

    process(engine, key, unavailable, tensor_enabled=True)
    with Session(engine) as db, db.begin():
        state = db.get(BuildState, key)
        attempt = state.attempt_id
        assert state.status == "building" and state.attempts == 1
        state.lease_until = state.due_at = datetime.now(UTC) - timedelta(seconds=1)
    process(engine, key, unavailable, tensor_enabled=True)
    with Session(engine) as db:
        state = db.get(BuildState, key)
        assert state.attempt_id == attempt and state.attempts == 1
