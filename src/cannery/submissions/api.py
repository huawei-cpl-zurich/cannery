"""Git-only, namespace-owned submission and revision requests/status."""

from typing import Annotated, Any
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from redis.exceptions import RedisError
from sqlalchemy import Engine, Select, select
from sqlalchemy.orm import Session

from cannery.git_intake import GitSource
from cannery.registry.api import identity
from cannery.registry.dispatch import publish_submission
from cannery.registry.freeze import kernel_key
from cannery.registry.manifest import Name
from cannery.registry.models import SpecVersion
from cannery.settings import Settings
from cannery.submissions.manifest import Device, Language
from cannery.submissions.models import Candidate, RequestedTarget, Submission, SubmissionConflict
from cannery.submissions.requests import new_submission, revise_submission
from cannery.submissions.state import IntakeState


class SubmitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    spec_version: int = Field(ge=1, strict=True)
    device: Device
    language: Language
    source: GitSource


class RevisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    commit: str = Field(pattern=r"^(?:[a-f0-9]{40}|[a-f0-9]{64})$")


def dispatch(settings: Settings, key: UUID) -> UUID:
    try:
        publish_submission(settings.redis_url.get_secret_value(), str(key))
    except RedisError:
        pass
    return key


def submit(
    settings: Settings, engine: Engine, namespace: str, name: str, body: SubmitRequest
) -> UUID:
    if namespace != identity(settings):
        raise HTTPException(403, "Namespace access denied")
    with Session(engine) as session:
        spec = session.scalar(
            select(SpecVersion).where(
                SpecVersion.kernel_id == kernel_key(namespace, name),
                SpecVersion.version == body.spec_version,
            )
        )
        if spec is None:
            raise HTTPException(404, "Frozen specification not found")
        spec_id = spec.id
    key = new_submission(
        engine,
        namespace,
        spec_id,
        body.source,
        RequestedTarget(device=body.device, language=body.language),
    )
    return dispatch(settings, key)


def revise(settings: Settings, engine: Engine, identifier: UUID, commit: str) -> UUID:
    namespace = identity(settings)
    try:
        key = revise_submission(engine, namespace, identifier, commit)
    except SubmissionConflict:
        raise HTTPException(404, "Retained parent submission not found") from None
    return dispatch(settings, key)


def query(namespace: str) -> Select[*tuple[Any, ...]]:
    return (
        select(Submission, IntakeState, Candidate, SpecVersion)
        .join(IntakeState, IntakeState.id == Submission.id)
        .outerjoin(Candidate, Candidate.id == Submission.id)
        .join(SpecVersion, SpecVersion.id == Submission.spec_id)
        .where(Submission.namespace == namespace)
    )


def describe(
    request: Submission, state: IntakeState, candidate: Candidate | None, spec: SpecVersion
) -> dict[str, Any]:
    return {
        "id": str(request.id),
        "parent_id": str(request.parent_id) if request.parent_id else None,
        "binding_id": str(candidate.binding_id) if candidate else None,
        "kernel": spec.manifest["kernel"],
        "spec_version": spec.version,
        "source": request.source,
        "target": request.target,
        "status": state.status,
        "error": state.error,
        "created_at": request.created_at.isoformat(),
        "eta_seconds": None,
        "stages": {
            "intake": "complete"
            if candidate
            else "failed"
            if state.status == "failed"
            else state.status,
            "build": "blocked" if state.status == "failed" else "pending",
            "correctness": "blocked" if state.status == "failed" else "pending",
            "benchmark": "blocked" if state.status == "failed" else "pending",
        },
        "candidate": None
        if candidate is None
        else {
            "content_sha256": candidate.content_sha256,
            "file_sha256": candidate.file_sha256,
            "manifest": candidate.manifest,
        },
    }


def get_info(session: Session, namespace: str, identifier: UUID) -> dict[str, Any]:
    row = session.execute(query(namespace).where(Submission.id == identifier)).one_or_none()
    if row is None:
        raise HTTPException(404, "Submission not found")
    return describe(row[0], row[1], row[2], row[3])


def list_info(session: Session, namespace: str) -> list[dict[str, Any]]:
    return [
        describe(row[0], row[1], row[2], row[3])
        for row in session.execute(
            query(namespace).order_by(Submission.created_at.desc()).limit(50)
        )
    ]


def install_submissions(app: FastAPI, settings: Settings, engine: Engine) -> None:
    def principal() -> str:
        return identity(settings)

    def accepted(key: UUID) -> dict[str, str]:
        return {"id": str(key), "status": "queued", "status_url": f"/api/v1/submissions/{key}"}

    @app.post("/api/v1/kernels/{namespace}/{name}/submissions", status_code=202)
    def create(
        namespace: Name, name: Name, body: SubmitRequest, user: Annotated[str, Depends(principal)]
    ) -> dict[str, str]:
        return accepted(submit(settings, engine, namespace, name, body))

    @app.post("/api/v1/submissions/{identifier}/revisions", status_code=202)
    def revision(
        identifier: UUID, body: RevisionRequest, user: Annotated[str, Depends(principal)]
    ) -> dict[str, str]:
        return accepted(revise(settings, engine, identifier, body.commit))

    @app.get("/api/v1/submissions/mine")
    def mine(user: Annotated[str, Depends(principal)]) -> list[dict[str, Any]]:
        with Session(engine) as session:
            return list_info(session, user)

    @app.get("/api/v1/submissions/{identifier}")
    def status(identifier: UUID, user: Annotated[str, Depends(principal)]) -> dict[str, Any]:
        with Session(engine) as session:
            return get_info(session, user, identifier)
