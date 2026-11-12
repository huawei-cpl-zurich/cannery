from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from test_build_state import candidate
from test_release_contract import parse
from test_release_contract import release_data as release_data
from test_submission_requests import records as records

from cannery.artifacts import digest
from cannery.releases.models import (
    Publication,
    PublicationEvent,
    PublicationJob,
    Release,
    ReleaseTolerance,
)


@pytest.fixture
def retained_release(records, release_data):
    engine, spec_id = records
    key = candidate(records)
    release_data.update(submission_id=str(key), spec_id=str(spec_id))
    document = parse(release_data)
    pin = dict(
        repository="private",
        revision=digest(document.encoded()),
        path="releases/manifest.json",
        sha256=digest(document.encoded()),
    )
    row = Release(
        id=uuid4(),
        manifest=pin,
        document=document.model_dump(mode="json"),
        score=999.0,
        cohort_sha256="incorrect",
    )
    with Session(engine) as db, db.begin():
        db.add(row)
        db.flush()
        identifier = row.id
        assert row.candidate_id == key and row.spec_id == spec_id
        assert row.score == 1.0 and row.cohort_sha256 == document.cohort_sha256
    return engine, identifier, key


def test_immutable_manifests_and_only_previously_tested_tolerances(retained_release):
    engine, release_id, _ = retained_release
    with Session(engine) as db, db.begin():
        db.add(ReleaseTolerance(release_id=release_id, rtol=1e-5, atol=1e-6))
    with Session(engine) as db:
        row = db.get(Release, release_id)
        row.score = 2.0
        with pytest.raises(ValueError, match="immutable"):
            db.commit()
    with Session(engine) as db:
        db.add(ReleaseTolerance(release_id=release_id, rtol=1.0, atol=1.0))
        with pytest.raises(ValueError, match="not validated"):
            db.commit()
    with Session(engine) as db:
        row = db.scalar(select(ReleaseTolerance))
        db.delete(row)
        with pytest.raises(ValueError, match="immutable"):
            db.commit()


def test_unverified_manifest_pin_cannot_create_release(records, release_data):
    engine, _ = records
    with Session(engine) as db:
        db.add(
            Release(
                manifest=dict(
                    repository="private", revision="a" * 64, sha256="a" * 64, path="manifest.json"
                ),
                document=release_data,
            )
        )
        with pytest.raises(ValueError, match="pin"):
            db.commit()


def test_visibility_requires_pinned_release_and_audit_is_immutable(retained_release):
    engine, release_id, key = retained_release
    actor = {"namespace": "team", "role": "user"}
    with Session(engine) as db, db.begin():
        db.add(Publication(candidate_id=key, status="publishing", generation=1, actor=actor))
    with Session(engine) as db:
        db.get(Publication, key).status = "published"
        with pytest.raises(IntegrityError):
            db.commit()
    with Session(engine) as db, db.begin():
        pub = db.get(Publication, key)
        pub.release_id = release_id
        pub.status = "published"
        pub.published_at = datetime.now(UTC)
        db.add(PublicationEvent(candidate_id=key, generation=1, kind="published", actor=actor))
    with Session(engine) as db:
        db.scalar(select(PublicationEvent)).kind = "withdrawn"
        with pytest.raises(ValueError, match="immutable"):
            db.commit()


def test_terminal_publication_job_is_immutable(retained_release):
    engine, _, key = retained_release
    with Session(engine) as db, db.begin():
        db.add(Publication(candidate_id=key, status="publishing", generation=1, actor={}))
        db.flush()
        job = PublicationJob(
            candidate_id=key,
            generation=1,
            status="queued",
            deadline=datetime.now(UTC) + timedelta(minutes=10),
        )
        db.add(job)
        db.flush()
        identifier = job.id
    with Session(engine) as db, db.begin():
        db.get(PublicationJob, identifier).status = "cancelled"
    with Session(engine) as db:
        db.get(PublicationJob, identifier).status = "queued"
        with pytest.raises(ValueError, match="immutable"):
            db.commit()


def test_publication_cannot_reference_another_candidates_release(retained_release):
    from sqlalchemy import insert

    from cannery.registry.models import SpecVersion

    engine, release_id, _ = retained_release
    with Session(engine) as db:
        spec_id = db.scalar(select(SpecVersion.id))
    other = candidate((engine, spec_id))
    with Session(engine) as db:
        db.add(
            Publication(
                candidate_id=other,
                release_id=release_id,
                status="published",
                generation=1,
                actor={},
                published_at=datetime.now(UTC),
            )
        )
        with pytest.raises(ValueError, match="own candidate"):
            db.commit()
    with engine.begin() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys=ON")
    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(
            insert(Publication).values(
                candidate_id=other,
                release_id=release_id,
                status="published",
                generation=1,
                actor={},
                published_at=datetime.now(UTC),
            )
        )
