from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session
from submission_fixtures import submission_files
from test_submission_requests import initial
from test_submission_requests import records as records

from cannery.artifacts import ArtifactRef
from cannery.builds.ascendc import BuildError
from cannery.builds.state import (
    BuildAttempt,
    BuildResult,
    BuildState,
    claim_build,
    current_build,
    due_builds,
    finish_build,
    heartbeat,
    queue_build,
)
from cannery.submissions.models import Candidate


def candidate(records, *, native=True):
    engine, _ = records
    key = initial(records, commit=uuid4().hex + "0" * 8)
    manifest = yaml.safe_load(submission_files("ascendc")["candidate/submission.yaml"])
    if native:
        manifest["implementation"]["device"] = "a5"
        manifest["implementation"]["build"]["abi"] = "unary-f32-v1"
    with Session(engine) as session, session.begin():
        frozen = Candidate(
            id=key,
            binding_id=key,
            binding={},
            manifest=manifest,
            snapshot={},
            content_sha256="a" * 64,
            file_sha256={},
        )
        session.add(frozen)
        session.flush()
        assert queue_build(session, frozen) == native
    return key


def expire(engine, key):
    with Session(engine) as session, session.begin():
        session.get(BuildState, key).lease_until = datetime.now(UTC) - timedelta(seconds=1)


def test_opt_in_and_recovery_retain_attempt_while_fencing_old_coordinator(records):
    engine, _ = records
    legacy = candidate(records, native=False)
    assert claim_build(engine, legacy) is None
    key = candidate(records)
    assert str(key) in due_builds(engine)
    first = claim_build(engine, key)
    assert first is not None and heartbeat(engine, first)
    assert claim_build(engine, key) is None
    expire(engine, key)
    assert not heartbeat(engine, first)
    second = claim_build(engine, key)
    assert second.attempt == first.attempt and second.token != first.token
    assert not finish_build(engine, first, error="worker_unavailable")
    with Session(engine) as session:
        assert current_build(session, second) is not None
        assert len(session.scalars(select(BuildAttempt)).all()) == 1
        assert session.get(BuildState, key).attempts == 1


def test_retry_backoff_budgets_and_immutable_results(records):
    engine, _ = records
    key = candidate(records)
    claims = []
    for _ in range(3):
        with Session(engine) as session, session.begin():
            session.get(BuildState, key).due_at = datetime.now(UTC)
        claim = claim_build(engine, key)
        claims.append(claim)
        assert finish_build(engine, claim, error="storage_unavailable", retryable=True)
        assert claim_build(engine, key) is None
    assert len({claim.attempt for claim in claims}) == 3
    with Session(engine) as session:
        assert session.get(BuildState, key).status == "failed"
        results = session.scalars(select(BuildResult)).all()
        assert len(results) == 3 and all(result.status == "failed" for result in results)
        results[0].error = "changed"
        with pytest.raises(BuildError, match="immutable"):
            session.commit()


def test_private_artifact_success_and_deadline_failure(records):
    engine, _ = records
    key = candidate(records)
    claim = claim_build(engine, key)
    reference = ArtifactRef(
        repository="private", revision="a" * 64, sha256="a" * 64, path="package.tar"
    )
    with pytest.raises(BuildError, match="pinned private"):
        finish_build(engine, claim, artifact=reference.model_copy(update={"repository": "fake"}))
    assert finish_build(engine, claim, artifact=reference)
    assert claim_build(engine, key) is None
    with Session(engine) as session:
        assert session.get(BuildState, key).status == "awaiting_validation"
        assert session.get(BuildResult, claim.attempt).artifact == reference.model_dump(mode="json")
    expired = candidate(records)
    pending = claim_build(engine, expired)
    expire(engine, expired)
    with Session(engine) as session, session.begin():
        session.get(BuildState, expired).deadline = datetime.now(UTC) - timedelta(seconds=1)
    assert claim_build(engine, expired) is None
    with Session(engine) as session:
        assert session.get(BuildState, expired).error == "deadline_exceeded"
        assert session.get(BuildResult, pending.attempt).error == "deadline_exceeded"
