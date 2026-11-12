from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session
from test_build_state import candidate
from test_release_contract import parse
from test_release_contract import release_data as release_data
from test_submission_requests import records as records

from cannery.artifacts import ArtifactRef, digest
from cannery.releases import state
from cannery.releases.models import Publication, PublicationEvent, PublicationJob, Release


@pytest.fixture
def publication(records, release_data, monkeypatch):
    engine, spec_id = records
    key = candidate(records)
    release_data.update(submission_id=str(key), spec_id=str(spec_id))
    manifest = parse(release_data)
    sha = digest(manifest.encoded())
    pin = ArtifactRef(repository="private", revision=sha, path="releases/manifest.json", sha256=sha)
    clock = [datetime.now(UTC) + timedelta(seconds=1)]
    monkeypatch.setattr(state, "now", lambda: clock[0])
    return engine, key, manifest, pin, clock


ACTOR = {"namespace": "team", "role": "user"}


def test_duplicate_publish_and_crash_recovery_preserve_release(publication):
    engine, key, manifest, pin, clock = publication
    job = state.request_publication(engine, key, ACTOR)
    assert state.request_publication(engine, key, ACTOR) == job and state.due_jobs(engine) == [job]
    first = state.claim_job(engine, job)
    assert first and state.heartbeat(engine, first)
    assert state.claim_job(engine, job) is None
    clock[0] += timedelta(seconds=91)
    second = state.claim_job(engine, job)
    assert second and second.token != first.token
    assert not state.complete(engine, first, manifest, pin)
    assert state.complete(engine, second, manifest, pin)
    assert not state.retry_job(engine, second, "late_error")
    assert state.request_publication(engine, key, ACTOR) == job
    with Session(engine) as db:
        pub = db.get(Publication, key)
        assert pub.status == "published"
        assert db.get(PublicationJob, job).attempts == 1
        assert len(list(db.scalars(select(Release)))) == 1
        assert [
            e.kind
            for e in db.scalars(select(PublicationEvent).order_by(PublicationEvent.created_at))
        ] == ["requested", "published"]


def test_withdrawal_fences_worker_and_republish_reuses_pin(publication):
    engine, key, manifest, pin, _ = publication
    job = state.request_publication(engine, key, ACTOR)
    claim = state.claim_job(engine, job)
    assert state.withdraw(engine, key, ACTOR) and state.withdraw(engine, key, ACTOR)
    assert not state.heartbeat(engine, claim) and not state.complete(engine, claim, manifest, pin)
    with Session(engine) as db:
        assert db.get(PublicationJob, job).status == "cancelled"
    next_job = state.request_publication(engine, key, ACTOR)
    assert next_job != job and state.complete(
        engine, state.claim_job(engine, next_job), manifest, pin
    )
    with Session(engine) as db:
        release_id = db.get(Publication, key).release_id
    assert state.withdraw(engine, key, ACTOR)
    third = state.request_publication(engine, key, ACTOR)
    assert state.complete(engine, state.claim_job(engine, third), manifest, pin)
    with Session(engine) as db:
        assert db.get(Publication, key).release_id == release_id
        assert len(list(db.scalars(select(Release)))) == 1


def test_outage_retries_are_bounded_and_backoff_cannot_be_skipped(publication):
    engine, key, _, _, clock = publication
    job = state.request_publication(engine, key, ACTOR)
    for number in range(1, 4):
        claim = state.claim_job(engine, job)
        assert claim
        assert state.retry_job(engine, claim, "artifact_storage_unavailable")
        assert state.claim_job(engine, job) is None
        clock[0] += timedelta(seconds=2**number + 1)
    with Session(engine) as db:
        assert db.get(Publication, key).status == "failed"
        row = db.get(PublicationJob, job)
        assert row.attempts == 3 and row.error == "artifact_storage_unavailable"
    assert state.due_jobs(engine) == []


def test_deadline_and_permanent_failure_never_make_release_visible(publication):
    engine, key, manifest, pin, clock = publication
    job = state.request_publication(engine, key, ACTOR)
    claim = state.claim_job(engine, job)
    clock[0] += timedelta(minutes=11)
    assert not state.complete(engine, claim, manifest, pin)
    assert state.claim_job(engine, job) is None
    with Session(engine) as db:
        assert db.get(PublicationJob, job).error == "deadline_exceeded"
    second = state.request_publication(engine, key, ACTOR)
    claim = state.claim_job(engine, second)
    assert state.retry_job(engine, claim, "invalid_release_evidence", permanent=True)
    with Session(engine) as db:
        assert db.get(Publication, key).release_id is None
        assert len(list(db.scalars(select(Release)))) == 0
