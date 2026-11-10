"""Durable registration state; Redis/Dramatiq messages carry IDs only."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import JSON, CheckConstraint, DateTime, Engine, ForeignKey, String, or_, select
from sqlalchemy.orm import Mapped, Session, mapped_column

import cannery.registry.models  # noqa: F401
from cannery.database import Base
from cannery.git_intake import GitSource


class Registration(Base):
    __tablename__ = "registrations"
    __table_args__ = (
        CheckConstraint("status IN ('queued','running','complete','failed')"),
        CheckConstraint("attempts BETWEEN 0 AND 3"),
        CheckConstraint("(status = 'complete') = (spec_id IS NOT NULL)"),
    )
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    namespace: Mapped[str] = mapped_column(String(64))
    source: Mapped[dict[str, Any]] = mapped_column(JSON)
    expected_name: Mapped[str | None] = mapped_column(String(64))
    allow_new_version: Mapped[bool]
    status: Mapped[str] = mapped_column(String(16), default="queued")
    attempts: Mapped[int] = mapped_column(default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    deadline: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    due_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    lease: Mapped[UUID | None]
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error: Mapped[str | None] = mapped_column(String(32))
    cleanup_handle: Mapped[str | None] = mapped_column(String(96))
    spec_id: Mapped[UUID | None] = mapped_column(ForeignKey("spec_versions.id"))


@dataclass(frozen=True)
class Claim:
    id: UUID
    token: UUID
    namespace: str
    source: GitSource
    expected_name: str | None
    allow_new_version: bool
    previous_handle: str | None


def aware(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


def create_registration(
    engine: Engine,
    source: GitSource,
    namespace: str,
    expected_name: str | None,
    allow_new_version: bool,
) -> UUID:
    now = datetime.now(UTC)
    with Session(engine, expire_on_commit=False) as session, session.begin():
        job = Registration(
            namespace=namespace,
            source=source.model_dump(mode="json"),
            expected_name=expected_name,
            allow_new_version=allow_new_version,
            created_at=now,
            deadline=now + timedelta(minutes=10),
            due_at=now,
        )
        session.add(job)
        session.flush()
        return job.id


def terminal(job: Registration, error: str) -> None:
    job.status, job.error, job.lease, job.lease_until = "failed", error, None, None


def claim_registration(engine: Engine, key: UUID, now: datetime | None = None) -> Claim | None:
    now = datetime.now(UTC) if now is None else now
    with Session(engine) as session, session.begin():
        job = session.scalar(
            select(Registration).where(Registration.id == key).with_for_update(skip_locked=True)
        )
        if job is None or job.status not in ("queued", "running"):
            return None
        if job.status == "running" and job.lease_until and aware(job.lease_until) > now:
            return None
        if aware(job.deadline) <= now or job.attempts >= 3:
            terminal(
                job, "deadline_exceeded" if aware(job.deadline) <= now else "attempts_exhausted"
            )
            return None
        if job.status == "queued" and aware(job.due_at) > now:
            return None
        old = job.cleanup_handle
        job.status, job.lease = "running", uuid4()
        job.lease_until = min(now + timedelta(seconds=180), aware(job.deadline))
        job.attempts += 1
        job.error = None
        return Claim(
            job.id,
            job.lease,
            job.namespace,
            GitSource.model_validate(job.source),
            job.expected_name,
            job.allow_new_version,
            old,
        )


def current_claim(session: Session, claim: Claim) -> Registration | None:
    job = session.get(Registration, claim.id, with_for_update=True)
    now = datetime.now(UTC)
    if (
        job is None
        or job.status != "running"
        or job.lease != claim.token
        or job.lease_until is None
        or aware(job.lease_until) <= now
        or aware(job.deadline) <= now
    ):
        return None
    return job


def fail_registration(engine: Engine, claim: Claim, code: str, retryable: bool) -> None:
    with Session(engine) as session, session.begin():
        job = current_claim(session, claim)
        if job is None:
            return
        if retryable and job.attempts < 3:
            job.status, job.error = "queued", code
            job.lease, job.lease_until = None, None
            job.due_at = datetime.now(UTC) + timedelta(seconds=5 * job.attempts)
        else:
            terminal(job, code)


def due_registrations(engine: Engine) -> list[str]:
    now = datetime.now(UTC)
    with Session(engine) as session, session.begin():
        active = list(
            session.scalars(
                select(Registration.id)
                .where(
                    Registration.status.in_(("queued", "running")),
                    or_(
                        Registration.deadline <= now,
                        (Registration.status == "queued") & (Registration.due_at <= now),
                        (Registration.status == "running") & (Registration.lease_until <= now),
                    ),
                )
                .order_by(Registration.created_at)
                .limit(50)
            )
        )
        cleanup = list(
            session.scalars(
                select(Registration)
                .where(
                    Registration.status.in_(("complete", "failed")),
                    Registration.cleanup_handle.is_not(None),
                    Registration.due_at <= now,
                )
                .order_by(Registration.due_at)
                .limit(50)
                .with_for_update(skip_locked=True)
            )
        )
        for job in cleanup:
            job.due_at = now + timedelta(seconds=30)
        return [str(key) for key in active] + [str(job.id) for job in cleanup]


def prepare_handle(engine: Engine, claim: Claim) -> str:
    from cannery.registry.sandbox import new_handle

    with Session(engine) as session, session.begin():
        job = current_claim(session, claim)
        if job is None or job.cleanup_handle is not None:
            raise ValueError("Attempt is stale or previous cleanup is pending")
        handle = new_handle()
        job.cleanup_handle = handle
        return handle


def clear_cleanup(engine: Engine, key: UUID, handle: str) -> None:
    with Session(engine) as session, session.begin():
        job = session.get(Registration, key, with_for_update=True)
        if job is not None and job.cleanup_handle == handle:
            job.cleanup_handle = None


def pending_cleanup(engine: Engine, key: UUID) -> str | None:
    with Session(engine) as session:
        job = session.get(Registration, key)
        if job is not None and job.status in ("complete", "failed"):
            return job.cleanup_handle
        return None
