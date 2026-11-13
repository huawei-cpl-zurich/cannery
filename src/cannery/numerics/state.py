"""Fenced numerical validation attempts, separate from binary admission."""

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
    or_,
    select,
)
from sqlalchemy.engine import Connection
from sqlalchemy.orm import Mapped, Mapper, Session, mapped_column

from cannery.database import Base, aware
from cannery.devices import Device
from cannery.submissions.models import Candidate


class NumericalResult(Base):
    __tablename__ = "numerical_results"
    __table_args__ = (
        UniqueConstraint("candidate_id", "number"),
        CheckConstraint("status IN ('passed','failed')"),
    )
    id: Mapped[UUID] = mapped_column(primary_key=True)
    candidate_id: Mapped[UUID] = mapped_column(ForeignKey("candidates.id"))
    number: Mapped[int]
    status: Mapped[str]
    report: Mapped[dict[str, Any] | None] = mapped_column(JSON(none_as_null=True))
    evidence: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    error: Mapped[str | None]
    finished_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class NumericalState(Base):
    __tablename__ = "submission_numerics"
    __table_args__ = (
        CheckConstraint("status IN ('queued','validating','passed','failed','blocked')"),
        CheckConstraint("attempts BETWEEN 0 AND 3"),
    )
    id: Mapped[UUID] = mapped_column(ForeignKey("candidates.id"), primary_key=True)
    baseline_key: Mapped[str | None] = mapped_column(ForeignKey("numerical_baselines.key"))
    status: Mapped[str] = mapped_column(default="queued")
    attempts: Mapped[int] = mapped_column(default=0)
    attempt_id: Mapped[UUID | None]
    lease: Mapped[UUID | None]
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    deadline: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC) + timedelta(minutes=30)
    )
    due_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    error: Mapped[str | None]


@event.listens_for(NumericalResult, "before_update")
@event.listens_for(NumericalResult, "before_delete")
def immutable(mapper: Mapper[Any], connection: Connection, target: NumericalResult) -> None:
    raise ValueError("Numerical results are immutable")


@dataclass(frozen=True)
class NumericalClaim:
    id: UUID
    attempt: UUID
    token: UUID


def _failure(session: Session, row: NumericalState, code: str) -> None:
    if row.attempt_id is not None and session.get(NumericalResult, row.attempt_id) is None:
        session.add(
            NumericalResult(
                id=row.attempt_id,
                candidate_id=row.id,
                number=row.attempts,
                status="failed",
                report=None,
                error=code,
                finished_at=datetime.now(UTC),
            )
        )
    row.status, row.error, row.lease, row.lease_until = "failed", code, None, None


def claim_numerical(engine: Engine, key: UUID, *, device: Device = "a5") -> NumericalClaim | None:
    now = datetime.now(UTC)
    with Session(engine) as session, session.begin():
        row = session.scalar(
            select(NumericalState)
            .join(Candidate, Candidate.id == NumericalState.id)
            .where(
                NumericalState.id == key,
                Candidate.manifest["implementation"]["device"].as_string() == device,
            )
            .with_for_update(of=NumericalState, skip_locked=True)
        )
        if row is None or row.status not in ("queued", "validating"):
            return None
        if row.lease_until is not None and aware(row.lease_until) > now:
            return None
        if aware(row.deadline) <= now:
            _failure(session, row, "deadline_exceeded")
            return None
        if row.status == "queued":
            if aware(row.due_at) > now:
                return None
            if row.attempts >= 3:
                _failure(session, row, "attempts_exhausted")
                return None
            row.attempts += 1
            row.attempt_id = uuid4()
        assert row.attempt_id is not None
        row.status, row.error, row.lease = "validating", None, uuid4()
        row.lease_until = min(aware(row.deadline), now + timedelta(seconds=90))
        return NumericalClaim(key, row.attempt_id, row.lease)


def current_numerical(session: Session, claim: NumericalClaim) -> NumericalState | None:
    row = session.get(NumericalState, claim.id, with_for_update=True)
    now = datetime.now(UTC)
    if (
        row is None
        or row.status != "validating"
        or row.lease != claim.token
        or row.attempt_id != claim.attempt
        or row.lease_until is None
        or aware(row.lease_until) <= now
        or aware(row.deadline) <= now
    ):
        return None
    return row


def heartbeat(engine: Engine, claim: NumericalClaim) -> bool:
    with Session(engine) as session, session.begin():
        row = current_numerical(session, claim)
        if row is None:
            return False
        row.lease_until = min(aware(row.deadline), datetime.now(UTC) + timedelta(seconds=90))
        return True


def postpone_numerical(engine: Engine, claim: NumericalClaim, delay: int = 5) -> bool:
    if not 1 <= delay <= 60:
        raise ValueError("Invalid numerical retry delay")
    with Session(engine) as session, session.begin():
        row = current_numerical(session, claim)
        if row is None:
            return False
        row.lease = None
        row.due_at = min(aware(row.deadline), datetime.now(UTC) + timedelta(seconds=delay))
        row.lease_until = row.due_at
        return True


def finish_numerical(
    engine: Engine,
    claim: NumericalClaim,
    *,
    report: dict[str, Any] | None = None,
    error: str | None = None,
    retryable: bool = False,
    blocked: bool = False,
    evidence: dict[str, Any] | None = None,
) -> bool:
    if (report is None) == (error is None):
        raise ValueError("Exactly one numerical report or error is required")
    if blocked and (report is not None or retryable):
        raise ValueError("Blocked validation requires a terminal error")
    if report is not None and type(report.get("passed")) is not bool:
        raise ValueError("Numerical report requires a boolean outcome")
    with Session(engine) as session, session.begin():
        row = current_numerical(session, claim)
        if row is None:
            return False
        passed = report is not None and report["passed"]
        session.add(
            NumericalResult(
                id=claim.attempt,
                candidate_id=claim.id,
                number=row.attempts,
                status="passed" if passed else "failed",
                report=report,
                error=error or (None if passed else "numerical_mismatch"),
                evidence=evidence or {},
                finished_at=datetime.now(UTC),
            )
        )
        row.status = (
            "passed"
            if passed
            else "queued"
            if retryable and error and row.attempts < 3
            else "failed"
        )
        if blocked:
            row.status = "blocked"
        row.error = error or (None if passed else "numerical_mismatch")
        row.lease = row.lease_until = None
        if row.status == "queued":
            row.due_at = datetime.now(UTC) + timedelta(seconds=5 * row.attempts)
        return True


def due_numericals(engine: Engine, *, device: Device = "a5") -> list[str]:
    now = datetime.now(UTC)
    with Session(engine) as session:
        return [
            str(key)
            for key in session.scalars(
                select(NumericalState.id)
                .join(Candidate, Candidate.id == NumericalState.id)
                .where(
                    Candidate.manifest["implementation"]["device"].as_string() == device,
                    NumericalState.status.in_(("queued", "validating")),
                    or_(
                        NumericalState.deadline <= now,
                        (NumericalState.status == "queued") & (NumericalState.due_at <= now),
                        (NumericalState.status == "validating")
                        & (NumericalState.lease_until <= now),
                    ),
                )
                .order_by(NumericalState.due_at)
                .limit(50)
            )
        ]
