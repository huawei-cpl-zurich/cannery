"""Durable publication intent; withdrawal fences every late worker commit."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import Engine, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from cannery.artifacts import ArtifactRef
from cannery.database import aware
from cannery.releases.contract import ReleaseManifest
from cannery.releases.models import (
    Publication,
    PublicationEvent,
    PublicationJob,
    Release,
    ReleaseTolerance,
)


def now() -> datetime:
    return datetime.now(UTC)


def record(db: Session, pub: Publication, kind: str, **details: Any) -> None:
    db.add(
        PublicationEvent(
            candidate_id=pub.candidate_id,
            generation=pub.generation,
            kind=kind,
            actor=dict(pub.actor),
            details=details,
        )
    )


def request_publication(engine: Engine, candidate: UUID, actor: dict[str, str]) -> UUID:
    # Concurrent first requests serialize on the unique publication candidate key.
    for retry in range(2):
        try:
            with Session(engine) as db, db.begin():
                pub = db.scalar(
                    select(Publication)
                    .where(Publication.candidate_id == candidate)
                    .with_for_update()
                )
                if pub is not None and pub.status in {"publishing", "published"}:
                    key = db.scalar(
                        select(PublicationJob.id).where(
                            PublicationJob.candidate_id == candidate,
                            PublicationJob.generation == pub.generation,
                        )
                    )
                    if key is None:
                        raise ValueError("Publication job missing")
                    return key
                if pub is None:
                    pub = Publication(
                        candidate_id=candidate, generation=1, status="publishing", actor=dict(actor)
                    )
                    db.add(pub)
                    db.flush()
                else:
                    pub.generation += 1
                    pub.status = "publishing"
                    pub.actor = dict(actor)
                job = PublicationJob(
                    id=uuid4(),
                    candidate_id=candidate,
                    generation=pub.generation,
                    deadline=now() + timedelta(minutes=10),
                )
                db.add(job)
                record(db, pub, "requested")
                return job.id
        except IntegrityError:
            if retry:
                raise
    raise AssertionError("Unreachable")


def withdraw(engine: Engine, candidate: UUID, actor: dict[str, str]) -> bool:
    with Session(engine) as db, db.begin():
        pub = db.scalar(
            select(Publication).where(Publication.candidate_id == candidate).with_for_update()
        )
        if pub is None:
            return False
        if pub.status == "withdrawn":
            return True
        job = db.scalar(
            select(PublicationJob)
            .where(
                PublicationJob.candidate_id == candidate,
                PublicationJob.generation == pub.generation,
            )
            .with_for_update()
        )
        if job is not None and job.status in {"queued", "running"}:
            job.status = "cancelled"
            job.lease = None
            job.lease_until = None
        pub.generation += 1
        pub.status = "withdrawn"
        pub.actor = dict(actor)
        record(db, pub, "withdrawn")
        return True


@dataclass(frozen=True)
class Claim:
    id: UUID
    candidate_id: UUID
    generation: int
    token: UUID


def locked(db: Session, identifier: UUID) -> tuple[Publication | None, PublicationJob | None]:
    # All paths lock publication before job, including withdrawal and heartbeat.
    candidate = db.scalar(
        select(PublicationJob.candidate_id).where(PublicationJob.id == identifier)
    )
    pub = db.scalar(
        select(Publication).where(Publication.candidate_id == candidate).with_for_update()
    )
    job = db.scalar(select(PublicationJob).where(PublicationJob.id == identifier).with_for_update())
    return pub, job


def fail(db: Session, pub: Publication, job: PublicationJob, error: str) -> None:
    job.status = "failed"
    job.error = error
    job.lease = None
    job.lease_until = None
    pub.status = "failed"
    record(db, pub, "failed", error=error, attempts=job.attempts)


def claim_job(engine: Engine, identifier: UUID) -> Claim | None:
    stamp = now()
    with Session(engine) as db, db.begin():
        pub, job = locked(db, identifier)
        if pub is None or job is None or job.status not in {"queued", "running"}:
            return None
        if pub.status != "publishing" or pub.generation != job.generation:
            job.status = "cancelled"
            job.lease = None
            job.lease_until = None
            return None
        if aware(job.deadline) <= stamp or job.status == "queued" and job.attempts >= 3:
            fail(
                db,
                pub,
                job,
                "deadline_exceeded" if aware(job.deadline) <= stamp else "attempts_exhausted",
            )
            return None
        if job.status == "queued" and aware(job.due_at) > stamp:
            return None
        if job.lease_until and aware(job.lease_until) > stamp:
            return None
        if job.status == "queued":
            job.attempts += 1
        token = uuid4()
        job.status = "running"
        job.lease = token
        job.lease_until = min(aware(job.deadline), stamp + timedelta(seconds=90))
        return Claim(job.id, job.candidate_id, job.generation, token)


def current(db: Session, claim: Claim) -> tuple[Publication, PublicationJob] | None:
    pub, job = locked(db, claim.id)
    if (
        pub is None
        or job is None
        or pub.status != "publishing"
        or pub.generation != claim.generation
        or job.candidate_id != claim.candidate_id
        or job.generation != claim.generation
        or job.status != "running"
        or job.lease != claim.token
        or job.lease_until is None
        or aware(job.lease_until) <= now()
        or aware(job.deadline) <= now()
    ):
        return None
    return pub, job


def heartbeat(engine: Engine, claim: Claim) -> bool:
    with Session(engine) as db, db.begin():
        owned = current(db, claim)
        if owned is None:
            return False
        _, job = owned
        job.lease_until = min(aware(job.deadline), now() + timedelta(seconds=90))
        return True


def retry_job(engine: Engine, claim: Claim, error: str, *, permanent: bool = False) -> bool:
    with Session(engine) as db, db.begin():
        owned = current(db, claim)
        if owned is None:
            return False
        pub, job = owned
        if permanent or job.attempts >= 3:
            fail(db, pub, job, error)
        else:
            job.status = "queued"
            job.error = error
            job.lease = None
            job.lease_until = None
            job.due_at = min(aware(job.deadline), now() + timedelta(seconds=2**job.attempts))
            record(db, pub, "retry", error=error, attempt=job.attempts)
        return True


def complete(engine: Engine, claim: Claim, manifest: ReleaseManifest, pin: ArtifactRef) -> bool:
    if manifest.submission_id != claim.candidate_id:
        raise ValueError("Publication candidate differs")
    with Session(engine) as db, db.begin():
        owned = current(db, claim)
        if owned is None:
            return False
        pub, job = owned
        release = db.scalar(select(Release).where(Release.candidate_id == claim.candidate_id))
        if release is None:
            release = Release(
                manifest=pin.model_dump(mode="json"), document=manifest.model_dump(mode="json")
            )
            db.add(release)
            db.flush()
            for t in manifest.tolerances:
                db.add(ReleaseTolerance(release_id=release.id, rtol=t.rtol, atol=t.atol))
        elif release.manifest != pin.model_dump(
            mode="json"
        ) or release.document != manifest.model_dump(mode="json"):
            raise ValueError("Immutable release evidence changed")
        pub.release_id = release.id
        pub.status = "published"
        pub.published_at = now()
        job.status = "complete"
        job.error = None
        job.lease = None
        job.lease_until = None
        record(db, pub, "published", release_id=str(release.id))
        return True


def due_jobs(engine: Engine) -> list[UUID]:
    stamp = now()
    with Session(engine) as db:
        return list(
            db.scalars(
                select(PublicationJob.id)
                .where(
                    PublicationJob.status.in_(["queued", "running"]),
                    PublicationJob.due_at <= stamp,
                    or_(
                        PublicationJob.lease_until.is_(None),
                        PublicationJob.lease_until <= stamp,
                        PublicationJob.deadline <= stamp,
                    ),
                )
                .order_by(PublicationJob.due_at, PublicationJob.id)
                .limit(50)
            )
        )
