"""Durable binary admission state; lease recovery preserves remote attempt identity."""

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

from cannery.artifacts import ArtifactRef
from cannery.builds.ascendc import AscendCAdapter, BuildError
from cannery.database import Base
from cannery.registry.registrations import aware
from cannery.submissions.manifest import SubmissionManifest
from cannery.submissions.models import Candidate


class BuildAttempt(Base):
    __tablename__ = "build_attempts"
    __table_args__ = (UniqueConstraint("candidate_id", "number"),)
    id: Mapped[UUID] = mapped_column(primary_key=True)
    candidate_id: Mapped[UUID] = mapped_column(ForeignKey("candidates.id"))
    number: Mapped[int]
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class BuildResult(Base):
    __tablename__ = "build_results"
    __table_args__ = (CheckConstraint("status IN ('complete','failed')"),)
    id: Mapped[UUID] = mapped_column(ForeignKey("build_attempts.id"), primary_key=True)
    status: Mapped[str]
    artifact: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    evidence: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    error: Mapped[str | None]
    finished_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class BuildState(Base):
    __tablename__ = "submission_builds"
    __table_args__ = (
        CheckConstraint("status IN ('queued','building','awaiting_validation','failed')"),
        CheckConstraint("attempts >= 0 AND attempts <= 3"),
    )
    id: Mapped[UUID] = mapped_column(ForeignKey("candidates.id"), primary_key=True)
    status: Mapped[str] = mapped_column(default="queued")
    attempts: Mapped[int] = mapped_column(default=0)
    attempt_id: Mapped[UUID | None] = mapped_column(ForeignKey("build_attempts.id"))
    deadline: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC) + timedelta(minutes=30),
    )
    due_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    lease: Mapped[UUID | None]
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error: Mapped[str | None]


@event.listens_for(BuildAttempt, "before_update")
@event.listens_for(BuildAttempt, "before_delete")
@event.listens_for(BuildResult, "before_update")
@event.listens_for(BuildResult, "before_delete")
def immutable(
    mapper: Mapper[Any], connection: Connection, target: BuildAttempt | BuildResult
) -> None:
    raise BuildError("Build attempts and results are immutable")


@dataclass(frozen=True)
class BuildClaim:
    id: UUID
    attempt: UUID
    token: UUID


def queue_build(session: Session, candidate: Candidate) -> bool:
    """Explicit ABI opt-in only; no author code is imported during discovery."""
    try:
        AscendCAdapter().preflight(SubmissionManifest.model_validate(candidate.manifest))
    except ValueError:
        return False
    if session.get(BuildState, candidate.id) is None:
        session.add(BuildState(id=candidate.id))
    return True


def _terminal(session: Session, state: BuildState, code: str, now: datetime) -> None:
    if state.attempt_id is not None and session.get(BuildResult, state.attempt_id) is None:
        session.add(
            BuildResult(
                id=state.attempt_id, status="failed", artifact=None, error=code, finished_at=now
            )
        )
    state.status, state.error = "failed", code
    state.lease, state.lease_until = None, None


def claim_build(engine: Engine, key: UUID) -> BuildClaim | None:
    now = datetime.now(UTC)
    with Session(engine) as session, session.begin():
        state = session.scalar(
            select(BuildState).where(BuildState.id == key).with_for_update(skip_locked=True)
        )
        if state is None or state.status not in ("queued", "building"):
            return None
        if state.status == "building" and state.lease_until and aware(state.lease_until) > now:
            return None
        if aware(state.deadline) <= now:
            _terminal(session, state, "deadline_exceeded", now)
            return None
        if state.status == "queued":
            if aware(state.due_at) > now:
                return None
            if state.attempts >= 3:
                _terminal(session, state, "attempts_exhausted", now)
                return None
            state.attempts += 1
            state.attempt_id = uuid4()
            session.add(
                BuildAttempt(
                    id=state.attempt_id, candidate_id=key, number=state.attempts, started_at=now
                )
            )
            session.flush()
        # Reclaiming an expired coordinator must resume the same retained operation.
        assert state.attempt_id is not None
        state.status, state.lease, state.error = "building", uuid4(), None
        state.lease_until = min(now + timedelta(seconds=90), aware(state.deadline))
        return BuildClaim(key, state.attempt_id, state.lease)


def current_build(session: Session, claim: BuildClaim) -> BuildState | None:
    state = session.get(BuildState, claim.id, with_for_update=True)
    now = datetime.now(UTC)
    if (
        state is None
        or state.status != "building"
        or state.lease != claim.token
        or state.attempt_id != claim.attempt
        or state.lease_until is None
        or aware(state.lease_until) <= now
        or aware(state.deadline) <= now
    ):
        return None
    return state


def heartbeat(engine: Engine, claim: BuildClaim) -> bool:
    with Session(engine) as session, session.begin():
        state = current_build(session, claim)
        if state is None:
            return False
        state.lease_until = min(datetime.now(UTC) + timedelta(seconds=90), aware(state.deadline))
        return True


def finish_build(
    engine: Engine,
    claim: BuildClaim,
    *,
    artifact: ArtifactRef | None = None,
    error: str | None = None,
    retryable: bool = False,
    evidence: dict[str, Any] | None = None,
) -> bool:
    if (artifact is None) == (error is None):
        raise BuildError("Exactly one artifact or error is required")
    if artifact is not None and (
        artifact.repository != "private" or artifact.revision != artifact.sha256
    ):
        raise BuildError("Build admission requires a pinned private artifact")
    with Session(engine) as session, session.begin():
        state = current_build(session, claim)
        if state is None:
            return False
        session.add(
            BuildResult(
                id=claim.attempt,
                status="complete" if artifact else "failed",
                artifact=artifact.model_dump(mode="json") if artifact else None,
                evidence=evidence or {},
                error=error,
                finished_at=datetime.now(UTC),
            )
        )
        state.status = (
            "awaiting_validation"
            if artifact
            else ("queued" if retryable and state.attempts < 3 else "failed")
        )
        state.error, state.lease, state.lease_until = error, None, None
        if state.status == "queued":
            state.due_at = datetime.now(UTC) + timedelta(seconds=5 * state.attempts)
        return True


def due_builds(engine: Engine) -> list[str]:
    now = datetime.now(UTC)
    with Session(engine) as session:
        return [
            str(key)
            for key in session.scalars(
                select(BuildState.id)
                .where(
                    BuildState.status.in_(("queued", "building")),
                    or_(
                        BuildState.deadline <= now,
                        (BuildState.status == "queued") & (BuildState.due_at <= now),
                        (BuildState.status == "building") & (BuildState.lease_until <= now),
                    ),
                )
                .order_by(BuildState.due_at)
                .limit(50)
            )
        ]
