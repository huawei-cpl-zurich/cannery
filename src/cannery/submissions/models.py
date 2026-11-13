"""Immutable request and retained-candidate envelopes; execution state is separate."""

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import JSON, CheckConstraint, DateTime, ForeignKey, Index, event
from sqlalchemy.engine import Connection
from sqlalchemy.orm import Mapped, Mapper, mapped_column

from cannery.database import Base
from cannery.registry.manifest import StrictModel
from cannery.submissions.manifest import Device, Language


class SubmissionConflict(ValueError):
    pass


class RequestedTarget(StrictModel):
    device: Device
    language: Language


class Submission(Base):
    __tablename__ = "submissions"
    __table_args__ = (
        CheckConstraint("parent_id <> id"),
        Index("ix_submissions_owner_time", "namespace", "created_at"),
    )
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    namespace: Mapped[str]
    spec_id: Mapped[UUID] = mapped_column(ForeignKey("spec_versions.id"))
    parent_id: Mapped[UUID | None] = mapped_column(ForeignKey("submissions.id"))
    source: Mapped[dict[str, Any]] = mapped_column(JSON)
    target: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class Candidate(Base):
    __tablename__ = "candidates"
    id: Mapped[UUID] = mapped_column(ForeignKey("submissions.id"), primary_key=True)
    binding_id: Mapped[UUID] = mapped_column(ForeignKey("submissions.id"))
    binding: Mapped[dict[str, Any]] = mapped_column(JSON)
    manifest: Mapped[dict[str, Any]] = mapped_column(JSON)
    snapshot: Mapped[dict[str, Any]] = mapped_column(JSON)
    content_sha256: Mapped[str]
    file_sha256: Mapped[dict[str, str]] = mapped_column(JSON)


class SubmissionReuse(Base):
    __tablename__ = "submission_reuse"
    __table_args__ = (CheckConstraint("id <> original_id"),)
    id: Mapped[UUID] = mapped_column(ForeignKey("candidates.id"), primary_key=True)
    original_id: Mapped[UUID] = mapped_column(ForeignKey("candidates.id"), index=True)


@event.listens_for(SubmissionReuse, "before_update")
@event.listens_for(SubmissionReuse, "before_delete")
@event.listens_for(Submission, "before_update")
@event.listens_for(Submission, "before_delete")
@event.listens_for(Candidate, "before_update")
@event.listens_for(Candidate, "before_delete")
def immutable(
    mapper: Mapper[Any], connection: Connection, target: Submission | Candidate | SubmissionReuse
) -> None:
    raise SubmissionConflict("Submission requests and accepted candidates are immutable")
