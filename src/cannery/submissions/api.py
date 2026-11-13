"""Git-only, namespace-owned submission and revision requests/status."""

import json
from typing import Annotated, Any
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from redis.exceptions import RedisError
from sqlalchemy import Engine, Select, func, select
from sqlalchemy.orm import Session

from cannery.accounts.context import identity
from cannery.artifacts import digest
from cannery.benchmark.state import PerformanceJob, PerformanceResult
from cannery.builds.state import BuildResult, BuildState
from cannery.git_intake import GitSource
from cannery.numerics.state import NumericalResult, NumericalState
from cannery.registry.dispatch import publish_submission
from cannery.registry.freeze import kernel_key
from cannery.registry.manifest import Name
from cannery.registry.models import SpecVersion
from cannery.releases.api import publication_info
from cannery.settings import Settings
from cannery.submissions.manifest import Device, Language
from cannery.submissions.models import (
    Candidate,
    RequestedTarget,
    Submission,
    SubmissionConflict,
    SubmissionReuse,
)
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
        select(
            Submission,
            IntakeState,
            Candidate,
            SpecVersion,
            BuildState,
            BuildResult,
            NumericalState,
            NumericalResult,
            PerformanceJob,
            PerformanceResult,
            SubmissionReuse.original_id,
        )
        .join(IntakeState, IntakeState.id == Submission.id)
        .outerjoin(Candidate, Candidate.id == Submission.id)
        .outerjoin(SubmissionReuse, SubmissionReuse.id == Candidate.id)
        .join(SpecVersion, SpecVersion.id == Submission.spec_id)
        .outerjoin(
            BuildState, BuildState.id == func.coalesce(SubmissionReuse.original_id, Candidate.id)
        )
        .outerjoin(BuildResult, BuildResult.id == BuildState.attempt_id)
        .outerjoin(
            NumericalState,
            NumericalState.id == func.coalesce(SubmissionReuse.original_id, Candidate.id),
        )
        .outerjoin(NumericalResult, NumericalResult.id == NumericalState.attempt_id)
        .outerjoin(
            PerformanceJob,
            PerformanceJob.candidate_id == func.coalesce(SubmissionReuse.original_id, Candidate.id),
        )
        .outerjoin(PerformanceResult, PerformanceResult.id == PerformanceJob.attempt)
        .where(Submission.namespace == namespace)
    )


def benchmark_info(
    job: PerformanceJob | None, result: PerformanceResult | None
) -> dict[str, Any] | None:
    if job is None:
        return None
    report = None
    if result and result.status == "ready" and result.report and result.report.get("passed"):
        report = {
            key: result.report[key]
            for key in ("protocol", "cases", "geomean_speedup")
            if key in result.report
        }
    return {
        "status": {"running": "benchmarking", "ready": "passed"}.get(job.status, job.status),
        "attempts": job.attempts,
        "error": job.error,
        "cohort_sha256": digest(json.dumps(job.identity["cohort"], sort_keys=True).encode()),
        "reference_key": job.reference_key,
        "report": report,
        "evidence_sha256": result.artifact["sha256"] if result and result.artifact else None,
    }


def describe(
    request: Submission,
    state: IntakeState,
    candidate: Candidate | None,
    spec: SpecVersion,
    build: BuildState | None = None,
    result: BuildResult | None = None,
    numerical: NumericalState | None = None,
    numerical_result: NumericalResult | None = None,
    performance: PerformanceJob | None = None,
    performance_result: PerformanceResult | None = None,
    reused_from: UUID | None = None,
) -> dict[str, Any]:
    failed = state.status == "failed" or (build is not None and build.status == "failed")
    status = (
        state.status
        if build is None
        else ("build_queued" if build.status == "queued" else build.status)
    )
    if numerical is not None:
        if numerical.status != "blocked":
            status = {"queued": "validation_queued", "passed": "awaiting_benchmark"}.get(
                numerical.status, numerical.status
            )
        failed = failed or numerical.status == "failed"
    benchmark = benchmark_info(performance, performance_result)
    if performance is not None:
        status = {
            "queued": "benchmark_queued",
            "running": "benchmarking",
            "ready": "complete",
            "blocked": "awaiting_benchmark",
            "failed": "failed",
        }[performance.status]
        failed = failed or performance.status == "failed"
    return {
        "id": str(request.id),
        "reused_from": str(reused_from) if reused_from else None,
        "parent_id": str(request.parent_id) if request.parent_id else None,
        "binding_id": str(candidate.binding_id) if candidate else None,
        "kernel": spec.manifest["kernel"],
        "spec_version": spec.version,
        "source": request.source,
        "target": request.target,
        "status": status,
        "error": performance.error
        if performance
        else numerical.error
        if numerical
        else build.error
        if build
        else state.error,
        "created_at": request.created_at.isoformat(),
        "eta_seconds": None,
        "stages": {
            "intake": "complete"
            if candidate
            else "failed"
            if state.status == "failed"
            else state.status,
            "build": ("complete" if build.status == "awaiting_validation" else build.status)
            if build
            else "blocked"
            if failed
            else "pending",
            "correctness": numerical.status if numerical else "blocked" if failed else "pending",
            "benchmark": benchmark["status"]
            if benchmark
            else "blocked"
            if failed or (numerical is not None and numerical.status == "blocked")
            else "pending",
        },
        "build": None
        if build is None
        else {
            "attempts": build.attempts,
            "artifact": result.artifact if result and result.status == "complete" else None,
        },
        "numerical": None
        if numerical is None
        else {
            "status": numerical.status,
            "attempts": numerical.attempts,
            "baseline_key": numerical.baseline_key,
            "report": numerical_result.report if numerical_result else None,
        },
        "benchmark": benchmark,
        "candidate": None
        if candidate is None
        else {
            "content_sha256": candidate.content_sha256,
            "file_sha256": candidate.file_sha256,
            "manifest": candidate.manifest,
        },
    }


def get_info(
    session: Session, namespace: str, identifier: UUID, *, publication: bool = False
) -> dict[str, Any]:
    row = session.execute(query(namespace).where(Submission.id == identifier)).one_or_none()
    if row is None:
        raise HTTPException(404, "Submission not found")
    result = describe(
        row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10]
    )
    if publication:
        result["publication"] = publication_info(session, row[10] or identifier)
    return result


def list_info(
    session: Session, namespace: str, *, publication: bool = False
) -> list[dict[str, Any]]:
    result = [
        describe(
            row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10]
        )
        for row in session.execute(
            query(namespace).order_by(Submission.created_at.desc()).limit(50)
        )
    ]
    if publication:
        for info in result:
            info["publication"] = publication_info(session, UUID(info["reused_from"] or info["id"]))
    return result


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
            return list_info(session, user, publication=settings.release_enabled)

    @app.get("/api/v1/submissions/{identifier}")
    def status(identifier: UUID, user: Annotated[str, Depends(principal)]) -> dict[str, Any]:
        with Session(engine) as session:
            return get_info(session, user, identifier, publication=settings.release_enabled)
