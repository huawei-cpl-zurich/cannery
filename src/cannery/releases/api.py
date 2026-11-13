"""Explicit namespace-owned publication actions; no native code in the API."""

from typing import Annotated, Any
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from cannery.accounts.context import identity, role
from cannery.database import aware
from cannery.registry.manifest import StrictModel
from cannery.releases.eligibility import PublicationUnavailable, publication_manifest
from cannery.releases.models import Publication, PublicationJob
from cannery.releases.state import request_publication, withdraw
from cannery.settings import Settings
from cannery.submissions.models import Submission, SubmissionReuse


class Action(StrictModel):
    pass


def owned(settings: Settings, db: Session, identifier: UUID) -> Submission:
    namespace = identity(settings)
    sub = db.get(Submission, identifier)
    if sub is None or sub.namespace != namespace and role(settings) != "admin":
        raise HTTPException(404, "Submission not found")
    reuse = db.get(SubmissionReuse, identifier)
    if reuse is not None:
        original = db.get(Submission, reuse.original_id)
        if (
            original is None
            or original.namespace != sub.namespace
            or original.spec_id != sub.spec_id
        ):
            raise HTTPException(409, "Invalid retained submission reuse")
        return original
    return sub


def job_info(job: PublicationJob) -> dict[str, Any]:
    return {
        "id": str(job.id),
        "status": job.status,
        "attempts": job.attempts,
        "error": job.error,
        "deadline": aware(job.deadline).isoformat(),
    }


def publication_info(db: Session, identifier: UUID) -> dict[str, Any]:
    pub = db.get(Publication, identifier)
    if pub is None:
        return {"status": "private", "release_id": None, "job": None}
    job = db.scalar(
        select(PublicationJob).where(
            PublicationJob.candidate_id == identifier, PublicationJob.generation == pub.generation
        )
    )
    return {
        "status": pub.status,
        "release_id": str(pub.release_id) if pub.release_id else None,
        "published_at": aware(pub.published_at).isoformat() if pub.published_at else None,
        "job": job_info(job) if job else None,
    }


def publish(settings: Settings, engine: Engine, identifier: UUID) -> dict[str, str]:
    with Session(engine) as db:
        identifier = owned(settings, db, identifier).id
    try:
        publication_manifest(engine, identifier)
    except PublicationUnavailable as error:
        raise HTTPException(409, str(error)) from None
    job = request_publication(
        engine, identifier, {"namespace": identity(settings), "role": role(settings)}
    )
    return {
        "id": str(job),
        "status_url": f"/api/v1/publication-jobs/{job}",
        "publication_url": f"/api/v1/submissions/{identifier}/publication",
    }


def withdraw_publication(settings: Settings, engine: Engine, identifier: UUID) -> dict[str, Any]:
    with Session(engine) as db:
        identifier = owned(settings, db, identifier).id
    if not withdraw(engine, identifier, {"namespace": identity(settings), "role": role(settings)}):
        raise HTTPException(409, "No publication to withdraw")
    with Session(engine) as db:
        return publication_info(db, identifier)


def install_publications(app: FastAPI, settings: Settings, engine: Engine) -> None:
    if not settings.release_enabled:
        return

    def principal() -> str:
        return identity(settings)

    @app.post("/api/v1/submissions/{identifier}/publish", status_code=202)
    def publication(
        identifier: UUID, body: Action, user: Annotated[str, Depends(principal)]
    ) -> dict[str, str]:
        return publish(settings, engine, identifier)

    @app.post("/api/v1/submissions/{identifier}/withdraw")
    def withdrawal(
        identifier: UUID, body: Action, user: Annotated[str, Depends(principal)]
    ) -> dict[str, Any]:
        return withdraw_publication(settings, engine, identifier)

    @app.get("/api/v1/submissions/{identifier}/publication")
    def status(identifier: UUID, user: Annotated[str, Depends(principal)]) -> dict[str, Any]:
        with Session(engine) as db:
            identifier = owned(settings, db, identifier).id
            return publication_info(db, identifier)

    @app.get("/api/v1/publication-jobs/{identifier}")
    def job_status(identifier: UUID, user: Annotated[str, Depends(principal)]) -> dict[str, Any]:
        with Session(engine) as db:
            job = db.get(PublicationJob, identifier)
            if job is None:
                raise HTTPException(404, "Publication job not found")
            owned(settings, db, job.candidate_id)
            return job_info(job)
