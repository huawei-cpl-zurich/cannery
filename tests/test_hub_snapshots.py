import re

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session
from test_publication_state import publication as publication
from test_publication_state import records as records
from test_publication_state import release_data as release_data

from cannery.releases import state
from cannery.releases.details import detail
from cannery.releases.hub import backfill, descriptor, ensure_snapshot, revision
from cannery.releases.models import HubSnapshot, Publication, Release

ACTOR = {"namespace": "team", "role": "user"}


def publish(publication):
    engine, key, manifest, pin, _ = publication
    job = state.request_publication(engine, key, ACTOR)
    assert state.complete(engine, state.claim_job(engine, job), manifest, pin)
    return engine, key


def test_publication_creates_stable_snapshot_and_metadata(publication):
    engine, key = publish(publication)
    with Session(engine) as db, db.begin():
        release = db.scalar(select(Release))
        snapshot = db.get(HubSnapshot, release.id)
        assert re.fullmatch("[a-f0-9]{40}", snapshot.revision)
        assert ensure_snapshot(db, release).revision == snapshot.revision
        assert revision(descriptor(release)) == snapshot.revision
        info = detail(db, release, db.get(Publication, key))
        assert info["hub"] == {
            "repo_id": snapshot.descriptor["repo_id"],
            "revision": snapshot.revision,
            "release_revision": f"release-{release.id}",
        }
    assert backfill(engine) == 0


def test_backfill_is_idempotent_and_does_not_change_release(publication):
    engine, _ = publish(publication)
    # Model a pre-migration release using SQL; normal ORM removal is forbidden.
    with engine.begin() as connection:
        connection.execute(HubSnapshot.__table__.delete())
    with Session(engine) as db:
        original = db.scalar(select(Release)).document
    assert backfill(engine) == 1
    assert backfill(engine) == 0
    with Session(engine) as db:
        assert db.scalar(select(Release)).document == original
        assert len(list(db.scalars(select(HubSnapshot)))) == 1


def test_withdrawal_keeps_mapping_but_is_not_backfilled(publication):
    engine, key = publish(publication)
    assert state.withdraw(engine, key, ACTOR)
    assert backfill(engine) == 0
    with Session(engine) as db:
        assert db.scalar(select(HubSnapshot)) is not None


def test_snapshot_cannot_be_changed_or_deleted(publication):
    engine, _ = publish(publication)
    for remove in [False, True]:
        with Session(engine) as db:
            row = db.scalar(select(HubSnapshot))
            if remove:
                db.delete(row)
            else:
                row.revision = "f" * 40
            with pytest.raises(ValueError, match="immutable"):
                db.commit()


def test_revision_binds_every_identity_field(publication):
    engine, _ = publish(publication)
    with Session(engine) as db:
        document = descriptor(db.scalar(select(Release)))
    original = revision(document)
    for field in document:
        changed = document | {field: "different"}
        assert revision(changed) != original
