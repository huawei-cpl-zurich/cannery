from typing import Annotated, Any
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, Path
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
from redis.exceptions import RedisError
from sqlalchemy import Engine, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from cannery.git_intake import GitSource
from cannery.registry.dispatch import publish_registration
from cannery.registry.freeze import kernel_key
from cannery.registry.manifest import Name
from cannery.registry.models import SpecVersion
from cannery.registry.registrations import Registration, create_registration
from cannery.settings import Settings


class RegistrationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: GitSource
    namespace: Name | None = None
    name: Name | None = None


def identity(settings: Settings) -> str:
    if not settings.dev_identity_enabled:
        raise HTTPException(401, "Authentication required")
    return settings.dev_namespace


def submit(settings: Settings, engine: Engine, request: RegistrationRequest) -> UUID:
    namespace = identity(settings)
    if request.namespace is not None and request.namespace != namespace:
        raise HTTPException(403, "Namespace access denied")
    key = create_registration(
        engine, request.source, namespace, request.name, settings.dev_role == "admin"
    )
    try:
        publish_registration(settings.redis_url.get_secret_value(), str(key))
    except RedisError:
        pass  # PostgreSQL-owned reconciliation recovers a lost dispatch.
    return key


def spec_info(spec: SpecVersion) -> dict[str, Any]:
    return {
        "id": str(spec.id),
        "version": spec.version,
        "kernel": spec.manifest["kernel"],
        "manifest": spec.manifest,
        "content_sha256": spec.content_sha256,
        "file_sha256": spec.file_sha256,
        "source": spec.snapshot["source"],
        "smoke": {"assurance": "diagnostic", "case_responses": spec.smoke["cases"]},
    }


def job_info(job: Registration, session: Session) -> dict[str, Any]:
    result: dict[str, Any] = {"id": str(job.id), "status": job.status, "error": job.error}
    if job.spec_id is not None:
        frozen = session.get(SpecVersion, job.spec_id)
        assert frozen is not None
        result["spec"] = spec_info(frozen)
    return result


def install_registry(app: FastAPI, settings: Settings, engine: Engine) -> None:
    def principal() -> str:
        return identity(settings)

    @app.exception_handler(SQLAlchemyError)
    async def unavailable(request: Any, error: SQLAlchemyError) -> JSONResponse:
        return JSONResponse({"detail": "Registry temporarily unavailable"}, status_code=503)

    @app.post("/api/v1/kernels", status_code=202)
    def register(
        request: RegistrationRequest, user: Annotated[str, Depends(principal)]
    ) -> dict[str, str]:
        key = submit(settings, engine, request)
        return {"id": str(key), "status": "queued", "status_url": f"/api/v1/registrations/{key}"}

    @app.get("/api/v1/registrations")
    def registrations(user: Annotated[str, Depends(principal)]) -> list[dict[str, Any]]:
        with Session(engine) as session:
            jobs = session.scalars(
                select(Registration)
                .where(Registration.namespace == user)
                .order_by(Registration.created_at.desc())
                .limit(50)
            )
            return [job_info(job, session) for job in jobs]

    @app.get("/api/v1/registrations/{key}")
    def registration(key: UUID, user: Annotated[str, Depends(principal)]) -> dict[str, Any]:
        with Session(engine) as session:
            job = session.get(Registration, key)
            if job is None or job.namespace != user:
                raise HTTPException(404, "Registration not found")
            return job_info(job, session)

    @app.get("/api/v1/kernels/{namespace}/{name}/specs/{version}")
    def frozen_spec(
        namespace: Name,
        name: Name,
        version: Annotated[int, Path(ge=1)],
        user: Annotated[str, Depends(principal)],
    ) -> dict[str, Any]:
        if namespace != user:
            raise HTTPException(404, "Specification not found")
        with Session(engine) as session:
            frozen = session.scalar(
                select(SpecVersion).where(
                    SpecVersion.kernel_id == kernel_key(namespace, name),
                    SpecVersion.version == version,
                )
            )
            if frozen is None:
                raise HTTPException(404, "Specification not found")
            return spec_info(frozen)
