"""Resolve compatible published packages, preserving explicit immutable pins."""

import json
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import Field, field_validator
from sqlalchemy.orm import Session

from cannery.artifacts import digest
from cannery.registry.manifest import Manifest
from cannery.registry.models import SpecVersion
from cannery.releases import catalog
from cannery.releases.contract import Tolerance, canonical
from cannery.releases.models import Publication, Release
from cannery.submissions.manifest import Language, Specialization

Value = Annotated[str, Field(min_length=1, max_length=256)]


class ResolveRequest(catalog.Selection):
    runtime: dict[str, Value] = Field(min_length=6, max_length=6)
    language: Language | None = None
    shape: tuple[int, ...] | None = None
    dtype: Literal["float32"] = "float32"
    release_id: UUID | None = None

    @field_validator("runtime")
    @classmethod
    def fingerprint(cls, value: dict[str, str]) -> dict[str, str]:
        if set(value) != {"machine", "python", "torch", "torch_npu", "cann", "soc"}:
            raise ValueError("Supply the exact runtime fingerprint")
        return value

    @field_validator("shape", mode="before")
    @classmethod
    def shape_domain(cls, value: Any) -> tuple[int, ...] | None:
        return None if value is None else Specialization(shape=value, dtype="float32").shape

    @field_validator("release_id", mode="before")
    @classmethod
    def identifier(cls, value: Any) -> Any:
        return UUID(value) if isinstance(value, str) else value


def compatible(release: Release, request: ResolveRequest) -> bool:
    m = catalog.manifest_of(release)
    return (
        m.kernel == request.kernel
        and m.device == request.device
        and (request.spec_version is None or m.spec_version == request.spec_version)
        and (request.language is None or m.language == request.language)
        and (request.cohort is None or release.cohort_sha256 == request.cohort)
        and m.supports(request.runtime, request.shape)
    )


def resolve(db: Session, request: ResolveRequest) -> dict[str, Any]:
    if request.release_id is not None:
        release, pub = catalog.release_info(db, request.release_id)
        if not compatible(release, request):
            raise catalog.SelectionError("release_not_compatible", 409)
        m = catalog.manifest_of(release)
        frozen = db.get(SpecVersion, m.spec_id)
        if frozen is None:
            raise catalog.SelectionError("specification_not_found")
        defaults = Manifest.model_validate_json(json.dumps(frozen.manifest)).spec.validation
        pair = Tolerance(
            rtol=defaults.default_rtol if request.rtol is None else request.rtol,
            atol=defaults.default_atol if request.atol is None else request.atol,
        )
        if pair not in m.tolerances:
            raise catalog.SelectionError("tolerance_not_validated", 422)
        return catalog.describe(release, pub)
    _, query = catalog.context(db, request, runtime_sha256=digest(canonical(request.runtime)))
    query = query.where(Release.score.is_not(None))
    if request.language is not None:
        query = query.where(Release.language == request.language)
    for release, pub in db.execute(
        query.order_by(
            Release.score.desc(), Publication.published_at, Release.id
        ).execution_options(yield_per=50)
    ):
        if compatible(release, request):
            return catalog.describe(release, pub)
    raise catalog.SelectionError("compatible_published_release_not_found")
