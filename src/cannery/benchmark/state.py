"""Fenced, retained performance jobs shared by references and candidates."""

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    Engine,
    ForeignKey,
    UniqueConstraint,
    event,
    func,
    inspect,
    or_,
    select,
)
from sqlalchemy.engine import Connection
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Mapped, Mapper, Session, mapped_column

import cannery.registry.models  # noqa: F401
import cannery.submissions.models  # noqa: F401
from cannery.artifacts import ArtifactRef
from cannery.database import Base, aware
from cannery.devices import Device


class PerformanceJob(Base):
    __tablename__ = "performance_jobs"
    __table_args__ = (
        CheckConstraint("kind IN ('reference','candidate')"),
        CheckConstraint("status IN ('queued','running','ready','failed','blocked')"),
        CheckConstraint("attempts BETWEEN 0 AND 3"),
        CheckConstraint("(kind = 'candidate') = (candidate_id IS NOT NULL)"),
        CheckConstraint("(kind = 'candidate') = (reference_key IS NOT NULL)"),
    )
    key: Mapped[str] = mapped_column(primary_key=True)
    kind: Mapped[str]
    candidate_id: Mapped[UUID | None] = mapped_column(ForeignKey("candidates.id"), unique=True)
    reference_key: Mapped[str | None] = mapped_column(ForeignKey("performance_jobs.key"))
    identity: Mapped[dict[str, Any]] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(default="queued")
    attempts: Mapped[int] = mapped_column(default=0)
    attempt: Mapped[UUID | None]
    lease: Mapped[UUID | None]
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    deadline: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    due_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    error: Mapped[str | None]


class PerformanceResult(Base):
    __tablename__ = "performance_results"
    __table_args__ = (
        UniqueConstraint("job_key", "number"),
        CheckConstraint("status IN ('ready','failed','blocked')"),
        CheckConstraint("status != 'ready' OR (artifact IS NOT NULL AND report IS NOT NULL)"),
    )
    id: Mapped[UUID] = mapped_column(primary_key=True)
    job_key: Mapped[str] = mapped_column(ForeignKey("performance_jobs.key"))
    number: Mapped[int]
    status: Mapped[str]
    report: Mapped[dict[str, Any] | None] = mapped_column(JSON(none_as_null=True))
    artifact: Mapped[dict[str, Any] | None] = mapped_column(JSON(none_as_null=True))
    error: Mapped[str | None]
    finished_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


@event.listens_for(PerformanceResult, "before_update")
@event.listens_for(PerformanceResult, "before_delete")
def immutable_result(mapper: Mapper[Any], connection: Connection, row: PerformanceResult) -> None:
    raise ValueError("Performance results are immutable")


@event.listens_for(PerformanceJob, "before_update")
@event.listens_for(PerformanceJob, "before_delete")
def immutable_terminal(mapper: Mapper[Any], connection: Connection, row: PerformanceJob) -> None:
    history = inspect(row).attrs.status.history
    terminal = {"ready", "failed", "blocked"}
    if (row.status in terminal and not history.has_changes()) or terminal.intersection(
        history.deleted
    ):
        raise ValueError("Terminal performance jobs are immutable")


def ensure_job(
    engine: Engine,
    key: str,
    kind: str,
    identity: dict[str, Any],
    *,
    candidate_id: UUID | None = None,
    reference_key: str | None = None,
) -> None:
    if re.fullmatch(r"[a-f0-9]{64}", key) is None or kind not in {"reference", "candidate"}:
        raise ValueError("Invalid performance identity")
    if (kind == "candidate" and (candidate_id is None or reference_key is None)) or (
        kind == "reference" and (candidate_id is not None or reference_key is not None)
    ):
        raise ValueError("Candidate jobs require candidate and reference identities")
    now = datetime.now(UTC)
    try:
        with Session(engine) as db, db.begin():
            if db.get(PerformanceJob, key) is None:
                db.add(
                    PerformanceJob(
                        key=key,
                        kind=kind,
                        identity=identity,
                        candidate_id=candidate_id,
                        reference_key=reference_key,
                        deadline=now + timedelta(minutes=30),
                        due_at=now,
                    )
                )
    except IntegrityError:
        pass
    with Session(engine) as db:
        row = db.get(PerformanceJob, key)
        if row is None or (row.kind, row.identity, row.candidate_id, row.reference_key) != (
            kind,
            identity,
            candidate_id,
            reference_key,
        ):
            raise ValueError("Conflicting performance identity")


@dataclass(frozen=True)
class PerformanceClaim:
    key: str
    attempt: UUID
    token: UUID


def claim_job(engine: Engine, key: str, *, device: Device = "a5") -> PerformanceClaim | None:
    now = datetime.now(UTC)
    with Session(engine) as db, db.begin():
        row = db.scalar(
            select(PerformanceJob)
            .where(
                PerformanceJob.key == key,
                func.coalesce(PerformanceJob.identity["cohort"]["device"].as_string(), "a5")
                == device,
            )
            .with_for_update(skip_locked=True)
        )
        if row is None or row.status not in {"queued", "running"}:
            return None
        if row.lease_until is not None and aware(row.lease_until) > now:
            return None
        if aware(row.deadline) <= now or (row.status == "queued" and row.attempts >= 3):
            error = "deadline_exceeded" if aware(row.deadline) <= now else "attempts_exhausted"
            if row.attempt and db.get(PerformanceResult, row.attempt) is None:
                db.add(
                    PerformanceResult(
                        id=row.attempt,
                        job_key=key,
                        number=row.attempts,
                        status="failed",
                        error=error,
                        finished_at=now,
                    )
                )
            row.status, row.error, row.lease, row.lease_until = "failed", error, None, None
            return None
        if aware(row.due_at) > now:
            return None
        if row.status == "queued":
            row.attempts += 1
            row.attempt = uuid4()
        assert row.attempt is not None
        row.status, row.lease = "running", uuid4()
        row.lease_until = min(aware(row.deadline), now + timedelta(seconds=90))
        return PerformanceClaim(key, row.attempt, row.lease)


def current_job(db: Session, claim: PerformanceClaim) -> PerformanceJob | None:
    row = db.get(PerformanceJob, claim.key, with_for_update=True)
    now = datetime.now(UTC)
    if (
        row is None
        or row.status != "running"
        or row.attempt != claim.attempt
        or row.lease != claim.token
        or row.lease_until is None
        or aware(row.lease_until) <= now
        or aware(row.deadline) <= now
    ):
        return None
    return row


def heartbeat(engine: Engine, claim: PerformanceClaim) -> bool:
    with Session(engine) as db, db.begin():
        row = current_job(db, claim)
        if row is None:
            return False
        row.lease_until = min(aware(row.deadline), datetime.now(UTC) + timedelta(seconds=90))
        return True


def postpone_job(engine: Engine, claim: PerformanceClaim, *, reason: str | None = None) -> bool:
    with Session(engine) as db, db.begin():
        row = current_job(db, claim)
        if row is None:
            return False
        row.lease = None
        row.error = reason
        row.due_at = min(aware(row.deadline), datetime.now(UTC) + timedelta(seconds=5))
        row.lease_until = row.due_at
        return True


def finish_job(
    engine: Engine,
    claim: PerformanceClaim,
    *,
    report: dict[str, Any] | None = None,
    artifact: ArtifactRef | None = None,
    error: str | None = None,
    retryable: bool = False,
    blocked: bool = False,
) -> bool:
    if error is None and (report is None or artifact is None or blocked or retryable):
        raise ValueError("Success requires report and verified raw timing artifact")
    if artifact is not None and (
        artifact.repository != "private" or artifact.revision != artifact.sha256
    ):
        raise ValueError("Performance evidence requires a pinned private artifact")
    with Session(engine) as db, db.begin():
        row = current_job(db, claim)
        if row is None:
            return False
        terminal = "ready" if error is None else "blocked" if blocked else "failed"
        db.add(
            PerformanceResult(
                id=claim.attempt,
                job_key=claim.key,
                number=row.attempts,
                status=terminal,
                report=report,
                artifact=artifact.model_dump(mode="json") if artifact else None,
                error=error,
                finished_at=datetime.now(UTC),
            )
        )
        row.status = "queued" if retryable and not blocked and row.attempts < 3 else terminal
        row.error, row.lease, row.lease_until = error, None, None
        if row.status == "queued":
            row.due_at = datetime.now(UTC) + timedelta(seconds=5 * row.attempts)
        return True


def due_jobs(engine: Engine, *, device: Device = "a5") -> list[str]:
    now = datetime.now(UTC)
    with Session(engine) as db:
        return list(
            db.scalars(
                select(PerformanceJob.key)
                .where(
                    func.coalesce(PerformanceJob.identity["cohort"]["device"].as_string(), "a5")
                    == device,
                    PerformanceJob.status.in_(("queued", "running")),
                    PerformanceJob.due_at <= now,
                    or_(PerformanceJob.lease_until.is_(None), PerformanceJob.lease_until <= now),
                )
                .order_by(PerformanceJob.due_at, PerformanceJob.key)
                .limit(100)
            )
        )
