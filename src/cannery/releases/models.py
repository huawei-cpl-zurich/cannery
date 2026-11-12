"""Immutable release evidence and fenced, auditable publication visibility."""

import json
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import JSON, CheckConstraint, DateTime, ForeignKey, UniqueConstraint, event
from sqlalchemy.engine import Connection
from sqlalchemy.orm import Mapped, Mapper, mapped_column

import cannery.submissions.models  # noqa: F401
from cannery.artifacts import ArtifactRef, digest
from cannery.database import Base
from cannery.releases.contract import ReleaseManifest, canonical


def now() -> datetime:
    return datetime.now(UTC)


class Release(Base):
    __tablename__ = "releases"
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    candidate_id: Mapped[UUID] = mapped_column(ForeignKey("candidates.id"), unique=True)
    spec_id: Mapped[UUID] = mapped_column(ForeignKey("spec_versions.id"), index=True)
    device: Mapped[str]
    language: Mapped[str]
    cohort_sha256: Mapped[str] = mapped_column(index=True)
    runtime_sha256: Mapped[str] = mapped_column(index=True)
    score: Mapped[float | None]
    manifest: Mapped[dict[str, Any]] = mapped_column(JSON)
    document: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class ReleaseTolerance(Base):
    __tablename__ = "release_tolerances"
    release_id: Mapped[UUID] = mapped_column(ForeignKey("releases.id"), primary_key=True)
    rtol: Mapped[float] = mapped_column(primary_key=True)
    atol: Mapped[float] = mapped_column(primary_key=True)


class Publication(Base):
    __tablename__ = "publications"
    __table_args__ = (
        CheckConstraint("status IN ('publishing','published','withdrawn','failed')"),
        CheckConstraint("generation >= 1"),
        CheckConstraint(
            "status != 'published' OR (release_id IS NOT NULL AND published_at IS NOT NULL)"
        ),
    )
    candidate_id: Mapped[UUID] = mapped_column(ForeignKey("candidates.id"), primary_key=True)
    release_id: Mapped[UUID | None] = mapped_column(ForeignKey("releases.id"), unique=True)
    status: Mapped[str]
    generation: Mapped[int]
    actor: Mapped[dict[str, str]] = mapped_column(JSON)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class PublicationJob(Base):
    __tablename__ = "publication_jobs"
    __table_args__ = (
        UniqueConstraint("candidate_id", "generation"),
        CheckConstraint("status IN ('queued','running','complete','failed','cancelled')"),
        CheckConstraint("attempts BETWEEN 0 AND 3"),
    )
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    candidate_id: Mapped[UUID] = mapped_column(ForeignKey("publications.candidate_id"))
    generation: Mapped[int]
    status: Mapped[str] = mapped_column(default="queued")
    attempts: Mapped[int] = mapped_column(default=0)
    lease: Mapped[UUID | None]
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    deadline: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    due_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    error: Mapped[str | None]


class PublicationEvent(Base):
    __tablename__ = "publication_events"
    __table_args__ = (
        CheckConstraint("kind IN ('requested','published','withdrawn','retry','failed')"),
    )
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    candidate_id: Mapped[UUID] = mapped_column(ForeignKey("publications.candidate_id"))
    generation: Mapped[int]
    kind: Mapped[str]
    actor: Mapped[dict[str, str]] = mapped_column(JSON)
    details: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


@event.listens_for(Release, "before_insert")
def bind_release(mapper: Mapper[Any], connection: Connection, target: Release) -> None:
    manifest = ReleaseManifest.model_validate_json(json.dumps(target.document))
    ref = ArtifactRef.model_validate(target.manifest)
    if (
        ref.repository != "private"
        or ref.revision != ref.sha256
        or ref.sha256 != digest(manifest.encoded())
    ):
        raise ValueError("Release manifest is not a verified private pin")
    target.candidate_id, target.spec_id = manifest.submission_id, manifest.spec_id
    target.device, target.language = manifest.device, manifest.language
    target.cohort_sha256 = manifest.cohort_sha256
    target.runtime_sha256 = digest(canonical(manifest.cohort.runtime))
    target.score = manifest.score.geomean_speedup


@event.listens_for(Release, "before_update")
@event.listens_for(Release, "before_delete")
@event.listens_for(ReleaseTolerance, "before_update")
@event.listens_for(ReleaseTolerance, "before_delete")
@event.listens_for(PublicationEvent, "before_update")
@event.listens_for(PublicationEvent, "before_delete")
def immutable(mapper: Mapper[Any], connection: Connection, target: Any) -> None:
    raise ValueError("Release evidence and publication events are immutable")


@event.listens_for(PublicationJob, "before_update")
@event.listens_for(PublicationJob, "before_delete")
def immutable_terminal(mapper: Mapper[Any], connection: Connection, target: PublicationJob) -> None:
    from sqlalchemy import inspect

    history = inspect(target).attrs.status.history
    terminal = {"complete", "failed", "cancelled"}
    if (
        terminal.intersection(history.deleted)
        or target.status in terminal
        and not history.has_changes()
    ):
        raise ValueError("Terminal publication jobs are immutable")


@event.listens_for(ReleaseTolerance, "before_insert")
def validated_tolerance(
    mapper: Mapper[Any], connection: Connection, target: ReleaseTolerance
) -> None:
    from sqlalchemy import select

    document = connection.scalar(select(Release.document).where(Release.id == target.release_id))
    if document is None or not any(
        t["rtol"] == target.rtol and t["atol"] == target.atol for t in document["tolerances"]
    ):
        raise ValueError("Tolerance was not validated for this release")
