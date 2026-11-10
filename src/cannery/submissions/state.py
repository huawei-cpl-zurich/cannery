"""PostgreSQL-owned intake leases; Redis carries only submission identifiers."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy import CheckConstraint, DateTime, Engine, ForeignKey, String, or_, select
from sqlalchemy.orm import Mapped, Session, mapped_column

from cannery.database import Base
from cannery.registry.registrations import aware
from cannery.submissions.models import Submission


class IntakeState(Base):
    __tablename__ = "submission_intake"
    __table_args__ = (
        CheckConstraint("status IN ('queued','intaking','awaiting_build','failed')"),
        CheckConstraint("attempts >= 0"),
    )
    id: Mapped[UUID] = mapped_column(ForeignKey("submissions.id"), primary_key=True)
    status: Mapped[str] = mapped_column(String(24), default="queued")
    attempts: Mapped[int] = mapped_column(default=0)
    deadline: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC) + timedelta(minutes=10)
    )
    due_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    lease: Mapped[UUID | None]
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    scratch_token: Mapped[UUID | None]
    error: Mapped[str | None] = mapped_column(String(64))


@dataclass(frozen=True)
class IntakeClaim:
    id: UUID
    token: UUID
    previous_scratch: UUID | None


def terminal(job: IntakeState, code: str) -> None:
    job.status, job.error, job.lease, job.lease_until = "failed", code, None, None


def claim_intake(engine: Engine, key: UUID, now: datetime | None = None) -> IntakeClaim | None:
    now = datetime.now(UTC) if now is None else now
    with Session(engine) as session, session.begin():
        job = session.scalar(
            select(IntakeState).where(IntakeState.id == key).with_for_update(skip_locked=True)
        )
        if job is None or job.status not in ("queued", "intaking"):
            return None
        if job.status == "intaking" and job.lease_until and aware(job.lease_until) > now:
            return None
        if aware(job.deadline) <= now or job.attempts >= 3:
            terminal(
                job, "deadline_exceeded" if aware(job.deadline) <= now else "attempts_exhausted"
            )
            return None
        if job.status == "queued" and aware(job.due_at) > now:
            return None
        job.status, job.lease = "intaking", uuid4()
        job.lease_until = min(now + timedelta(seconds=180), aware(job.deadline))
        job.attempts += 1
        job.error = None
        return IntakeClaim(job.id, job.lease, job.scratch_token)


def current_intake(session: Session, claim: IntakeClaim) -> IntakeState | None:
    job = session.get(IntakeState, claim.id, with_for_update=True)
    now = datetime.now(UTC)
    if (
        job is None
        or job.status != "intaking"
        or job.lease != claim.token
        or job.lease_until is None
        or aware(job.lease_until) <= now
        or aware(job.deadline) <= now
    ):
        return None
    return job


def fail_intake(engine: Engine, claim: IntakeClaim, code: str, retryable: bool) -> None:
    with Session(engine) as session, session.begin():
        job = current_intake(session, claim)
        if job is None:
            return
        if retryable and job.attempts < 3:
            job.status, job.error, job.lease, job.lease_until = "queued", code, None, None
            job.due_at = datetime.now(UTC) + timedelta(seconds=5 * job.attempts)
        else:
            terminal(job, code)


def prepare_scratch(engine: Engine, claim: IntakeClaim) -> UUID:
    with Session(engine) as session, session.begin():
        job = current_intake(session, claim)
        if job is None or job.scratch_token is not None:
            raise ValueError("Attempt is stale or prior scratch cleanup is pending")
        job.scratch_token = claim.token
        return claim.token


def clear_scratch(engine: Engine, key: UUID, token: UUID) -> None:
    with Session(engine) as session, session.begin():
        job = session.get(IntakeState, key, with_for_update=True)
        if job is not None and job.scratch_token == token:
            job.scratch_token = None


def pending_scratch(engine: Engine, key: UUID) -> UUID | None:
    with Session(engine) as session:
        job = session.get(IntakeState, key)
        if job is not None and job.status in ("awaiting_build", "failed"):
            return job.scratch_token
        return None


def due_intakes(engine: Engine) -> list[str]:
    now = datetime.now(UTC)
    with Session(engine) as session, session.begin():
        active = list(
            session.scalars(
                select(IntakeState.id)
                .join(Submission)
                .where(
                    IntakeState.status.in_(("queued", "intaking")),
                    or_(
                        IntakeState.deadline <= now,
                        (IntakeState.status == "queued") & (IntakeState.due_at <= now),
                        (IntakeState.status == "intaking") & (IntakeState.lease_until <= now),
                    ),
                )
                .order_by(Submission.created_at)
                .limit(50)
            )
        )
        cleanup = list(
            session.scalars(
                select(IntakeState)
                .where(
                    IntakeState.status.in_(("awaiting_build", "failed")),
                    IntakeState.scratch_token.is_not(None),
                    IntakeState.due_at <= now,
                )
                .order_by(IntakeState.due_at)
                .limit(50)
                .with_for_update(skip_locked=True)
            )
        )
        for job in cleanup:
            job.due_at = now + timedelta(seconds=30)
        return [str(key) for key in active] + [str(job.id) for job in cleanup]
