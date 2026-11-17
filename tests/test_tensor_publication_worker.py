from datetime import UTC, datetime, timedelta

import pytest
import test_publication_worker as legacy
from release_fixtures import admission as admission
from release_fixtures import arguments as arguments
from release_fixtures import benchmark as benchmark
from release_fixtures import records as records
from release_fixtures import setup_executor as setup_executor
from sqlalchemy.orm import Session
from test_release_contract import release_data as release_data
from test_tensor_release_assets import assets as assets
from test_tensor_release_contract import tensor_release as tensor_release

from cannery.releases import state, worker
from cannery.releases.eligibility import PublicationUnavailable, publication_document
from cannery.releases.models import Publication, PublicationJob, Release
from cannery.submissions.models import Submission


@pytest.fixture
def publication(admission, assets, monkeypatch):
    engine, key, _, _ = admission
    store, manifest, _, _ = assets
    with Session(engine) as db:
        spec_id = db.get(Submission, key).spec_id
    manifest = type(manifest).model_validate(
        manifest.model_dump() | {"submission_id": key, "spec_id": spec_id}
    )
    monkeypatch.setattr(worker, "publication_document", lambda *args: manifest)
    clock = [datetime.now(UTC) + timedelta(seconds=1)]
    monkeypatch.setattr(state, "now", lambda: clock[0])
    job = state.request_publication(engine, key, legacy.ACTOR)
    return engine, key, store, job, clock


@pytest.mark.parametrize(
    "check",
    [
        legacy.test_storage_outage_retries_without_partial_visibility,
        legacy.test_lost_database_reply_after_upload_resumes_same_attempt,
        legacy.test_withdraw_after_upload_fences_visibility_then_republish_reuses_bytes,
        legacy.test_lost_reply_after_commit_does_not_publish_twice,
    ],
)
def test_tensor_publication_preserves_shared_failure_and_fencing_guarantees(
    publication, monkeypatch, check
):
    check(publication, monkeypatch)


def test_tensor_publication_is_visible_with_exact_dispatch_and_idempotent_blobs(publication):
    engine, key, store, job, _ = publication
    worker.process(engine, job, store)
    with Session(engine) as db:
        release = db.get(Release, db.get(Publication, key).release_id)
        assert release.document["schema_version"] == 2
        assert release.document["cases"]["small"]["aliases"] == {"updated": "cache"}
        assert db.get(PublicationJob, job).status == "complete"
    puts = store.puts
    worker.process(engine, job, store)
    assert store.puts == puts


def test_snapshot_disconnect_retries_instead_of_permanently_failing(publication, monkeypatch):
    engine, key, store, job, clock = publication
    original = worker.publication_document

    def unavailable(*args):
        raise PublicationUnavailable("release_snapshot_unavailable")

    monkeypatch.setattr(worker, "publication_document", unavailable)
    worker.process(engine, job, store)
    with Session(engine) as db:
        assert db.get(PublicationJob, job).status == "queued"
        assert db.query(Release).count() == 0
    monkeypatch.setattr(worker, "publication_document", original)
    clock[0] += timedelta(seconds=3)
    worker.process(engine, job, store)
    with Session(engine) as db:
        assert db.get(Publication, key).status == "published"


def test_publication_dispatch_preserves_unary_without_snapshot_dependency(admission):
    engine, key, _, _ = admission
    assert publication_document(engine, key).schema_version == 1
