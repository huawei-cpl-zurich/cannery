"""Published-only discovery and rankings; never compare different cohorts."""

import json
from typing import Any, Self
from uuid import UUID

from pydantic import Field, model_validator
from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from cannery.database import aware
from cannery.devices import Device
from cannery.registry.manifest import Manifest, StrictModel
from cannery.registry.models import Kernel, SpecVersion
from cannery.releases.contract import SHA, ReleaseManifest, Tolerance
from cannery.releases.models import Publication, Release, ReleaseTolerance
from cannery.submissions.manifest import KernelPath


class Selection(StrictModel):
    kernel: KernelPath
    spec_version: int | None = Field(default=None, ge=1)
    device: Device = "a5"
    cohort: str | None = Field(default=None, pattern=SHA)
    rtol: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    atol: float | None = Field(default=None, ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def pair(self) -> Self:
        if (self.rtol is None) != (self.atol is None):
            raise ValueError("Supply both rtol and atol")
        return self


class SelectionError(ValueError):
    def __init__(self, code: str, status: int = 404, *, cohorts: list[str] | None = None):
        super().__init__(code)
        self.status = status
        self.cohorts = cohorts


def published() -> Select[*tuple[Release, Publication]]:
    return (
        select(Release, Publication)
        .join(Publication, Publication.release_id == Release.id)
        .where(Publication.status == "published", Publication.candidate_id == Release.candidate_id)
    )


def manifest_of(release: Release) -> ReleaseManifest:
    return ReleaseManifest.model_validate_json(json.dumps(release.document))


def describe(release: Release, pub: Publication) -> dict[str, Any]:
    m = manifest_of(release)
    assert pub.published_at is not None
    return {
        "id": str(release.id),
        "submission_id": str(release.candidate_id),
        "kernel": m.kernel,
        "spec_version": m.spec_version,
        "spec_sha256": m.spec_sha256,
        "device": release.device,
        "language": release.language,
        "cohort": release.cohort_sha256,
        "runtime": m.cohort.runtime,
        "abi": m.abi,
        "specializations": [s.model_dump(mode="json") for s in m.specializations]
        if m.specializations
        else None,
        "tolerances": [t.model_dump() for t in m.tolerances],
        "score": m.score.model_dump(mode="json"),
        "source": m.source.model_dump(mode="json"),
        "source_sha256": m.source_sha256,
        "manifest_sha256": release.manifest["sha256"],
        "package_sha256": m.package.sha256,
        "published_at": aware(pub.published_at).isoformat(),
        "publisher": pub.actor["namespace"],
    }


def release_info(db: Session, identifier: UUID) -> tuple[Release, Publication]:
    row = db.execute(published().where(Release.id == identifier)).one_or_none()
    if row is None:
        raise SelectionError("published_release_not_found")
    return row[0], row[1]


def kernels(db: Session) -> list[dict[str, Any]]:
    rows = db.execute(
        select(Kernel.namespace, Kernel.name, SpecVersion.version, Release.device)
        .select_from(Release)
        .join(Publication, Publication.release_id == Release.id)
        .join(SpecVersion, SpecVersion.id == Release.spec_id)
        .join(Kernel, Kernel.id == SpecVersion.kernel_id)
        .where(Publication.status == "published", Publication.candidate_id == Release.candidate_id)
        .distinct()
    ).all()
    grouped: dict[str, dict[str, Any]] = {}
    for namespace, name, version, device in rows:
        key = f"{namespace}/{name}"
        entry = grouped.setdefault(key, {"kernel": key, "versions": set(), "devices": set()})
        entry["versions"].add(version)
        entry["devices"].add(device)
    return [
        {
            "kernel": key,
            "versions": sorted(value["versions"], reverse=True),
            "devices": sorted(value["devices"]),
        }
        for key, value in sorted(grouped.items())
    ]


def context(
    db: Session, selection: Selection, *, runtime_sha256: str | None = None
) -> tuple[dict[str, Any], Select[*tuple[Release, Publication]]]:
    namespace, name = selection.kernel.split("/")
    query = (
        select(SpecVersion)
        .join(Kernel, Kernel.id == SpecVersion.kernel_id)
        .where(Kernel.namespace == namespace, Kernel.name == name)
    )
    if selection.spec_version is not None:
        query = query.where(SpecVersion.version == selection.spec_version)
    frozen = db.scalar(query.order_by(SpecVersion.version.desc()).limit(1))
    if frozen is None:
        raise SelectionError("specification_not_found")
    base = published().where(Release.spec_id == frozen.id, Release.device == selection.device)
    if runtime_sha256 is not None:
        base = base.where(Release.runtime_sha256 == runtime_sha256)
    cohorts = sorted(set(db.scalars(base.with_only_columns(Release.cohort_sha256))))
    if not cohorts:
        raise SelectionError("no_published_release")
    if selection.cohort is None and len(cohorts) != 1:
        raise SelectionError("cohort_required", 409, cohorts=cohorts)
    chosen = selection.cohort or cohorts[0]
    if chosen not in cohorts:
        raise SelectionError("published_cohort_not_found")
    validation = Manifest.model_validate_json(json.dumps(frozen.manifest)).spec.validation
    pair = Tolerance(
        rtol=validation.default_rtol if selection.rtol is None else selection.rtol,
        atol=validation.default_atol if selection.atol is None else selection.atol,
    )
    base = base.where(Release.cohort_sha256 == chosen)
    matched = base.join(ReleaseTolerance, ReleaseTolerance.release_id == Release.id).where(
        ReleaseTolerance.rtol == pair.rtol, ReleaseTolerance.atol == pair.atol
    )
    if db.scalar(select(matched.exists())) is not True:
        raise SelectionError("tolerance_not_validated", 422)
    info = {
        "kernel": selection.kernel,
        "spec_version": frozen.version,
        "device": selection.device,
        "cohort": chosen,
        "cohorts": cohorts,
        "tolerance": pair.model_dump(),
    }
    return info, matched


def leaderboard(
    db: Session, selection: Selection, limit: int = 50, offset: int = 0
) -> dict[str, Any]:
    info, query = context(db, selection)
    query = query.where(Release.score.is_not(None))
    total = db.scalar(select(func.count()).select_from(query.subquery()))
    rows = db.execute(
        query.order_by(Release.score.desc(), Publication.published_at, Release.id)
        .limit(limit)
        .offset(offset)
    )
    return info | {
        "rows": [describe(r, p) for r, p in rows],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


def listing(db: Session, selection: Selection, limit: int = 50, offset: int = 0) -> dict[str, Any]:
    info, query = context(db, selection)
    total = db.scalar(select(func.count()).select_from(query.subquery()))
    rows = db.execute(
        query.order_by(Release.score.desc().nulls_last(), Publication.published_at, Release.id)
        .limit(limit)
        .offset(offset)
    )
    return info | {
        "rows": [describe(r, p) for r, p in rows],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


def history(db: Session, selection: Selection) -> dict[str, Any]:
    info, query = context(db, selection)
    best = 0.0
    points = []
    for release, pub in db.execute(
        query.where(Release.score.is_not(None)).order_by(Publication.published_at, Release.id)
    ):
        assert release.score is not None and pub.published_at is not None
        if release.score > best:
            best = release.score
            points.append(
                {
                    "release_id": str(release.id),
                    "published_at": aware(pub.published_at).isoformat(),
                    "geomean_speedup": best,
                    "language": release.language,
                }
            )
    return info | {"points": points}
