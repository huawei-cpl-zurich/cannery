import copy
from datetime import UTC, datetime, timedelta

import pytest
from release_fixtures import admission as admission
from release_fixtures import arguments as arguments
from release_fixtures import benchmark as benchmark
from release_fixtures import records as records
from release_fixtures import setup_executor as setup_executor
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from cannery.artifacts import ArtifactError
from cannery.releases import state, worker
from cannery.releases.contract import read_manifest
from cannery.releases.eligibility import publication_manifest
from cannery.releases.models import Publication, PublicationJob, Release

ACTOR = {"namespace": "team", "role": "user"}


@pytest.fixture
def publication(admission, monkeypatch):
    engine, key, store, _ = admission
    clock = [datetime.now(UTC) + timedelta(seconds=1)]
    monkeypatch.setattr(state, "now", lambda: clock[0])
    job = state.request_publication(engine, key, ACTOR)
    return engine, key, store, job, clock


def test_publication_stores_verified_blobs_before_visibility_and_is_idempotent(publication):
    engine, key, store, job, _ = publication
    worker.process(engine, job, store)
    with Session(engine) as db:
        pub = db.get(Publication, key)
        assert pub.status == "published"
        release = db.get(Release, pub.release_id)
        ref = release.manifest
        manifest = read_manifest(store.data[ref["sha256"]], ref["sha256"])
        assert manifest.package.path.startswith("releases/")
        assert db.get(PublicationJob, job).status == "complete"
    puts = store.puts
    worker.process(engine, job, store)
    assert store.puts == puts


def test_storage_outage_retries_without_partial_visibility(publication, monkeypatch):
    engine, key, store, job, clock = publication
    original = store.get

    def offline(_):
        raise ArtifactError("offline")

    monkeypatch.setattr(store, "get", offline)
    worker.process(engine, job, store)
    with Session(engine) as db:
        assert db.get(Publication, key).status == "publishing"
        assert db.get(PublicationJob, job).status == "queued" and db.query(Release).count() == 0
    monkeypatch.setattr(store, "get", original)
    clock[0] += timedelta(seconds=3)
    worker.process(engine, job, store)
    with Session(engine) as db:
        assert db.get(Publication, key).status == "published"


def test_lost_database_reply_after_upload_resumes_same_attempt(publication, monkeypatch):
    engine, key, store, job, clock = publication
    original = worker.complete

    def interrupted(*args):
        raise SQLAlchemyError("disconnect before commit")

    monkeypatch.setattr(worker, "complete", interrupted)
    worker.process(engine, job, store)
    saved = copy.deepcopy(store.data)
    with Session(engine) as db:
        assert db.get(Publication, key).status == "publishing"
    monkeypatch.setattr(worker, "complete", original)
    clock[0] += timedelta(seconds=91)
    worker.process(engine, job, store)
    with Session(engine) as db:
        assert db.get(Publication, key).status == "published" and db.query(Release).count() == 1
        assert db.get(PublicationJob, job).attempts == 1
    assert store.data == saved


def test_withdraw_after_upload_fences_visibility_then_republish_reuses_bytes(
    publication, monkeypatch
):
    engine, key, store, job, _ = publication
    original = store.put

    def withdraw(path, raw):
        ref = original(path, raw)
        if path.endswith("manifest.json"):
            state.withdraw(engine, key, ACTOR)
        return ref

    monkeypatch.setattr(store, "put", withdraw)
    worker.process(engine, job, store)
    with Session(engine) as db:
        assert db.get(Publication, key).status == "withdrawn" and db.query(Release).count() == 0
    saved = copy.deepcopy(store.data)
    monkeypatch.setattr(store, "put", original)
    second = state.request_publication(engine, key, ACTOR)
    worker.process(engine, second, store)
    with Session(engine) as db:
        assert db.get(Publication, key).status == "published"
    assert store.data == saved


def test_corrupted_binary_fails_publication_but_preserves_benchmark(publication):
    engine, key, store, job, _ = publication
    manifest = publication_manifest(engine, key)
    store.data[manifest.package.sha256] = b"bad"
    worker.process(engine, job, store)
    with Session(engine) as db:
        assert db.get(Publication, key).status == "failed" and db.query(Release).count() == 0
        assert db.get(PublicationJob, job).error == "invalid_release_artifact"
    assert publication_manifest(engine, key).score.geomean_speedup == 0.5


def test_lost_reply_after_commit_does_not_publish_twice(publication, monkeypatch):
    engine, key, store, job, _ = publication
    original = worker.complete

    def lost_reply(*args):
        original(*args)
        raise SQLAlchemyError("reply lost after commit")

    monkeypatch.setattr(worker, "complete", lost_reply)
    worker.process(engine, job, store)
    puts = store.puts
    worker.process(engine, job, store)
    assert store.puts == puts
    with Session(engine) as db:
        assert db.get(Publication, key).status == "published"
        assert db.query(Release).count() == 1
