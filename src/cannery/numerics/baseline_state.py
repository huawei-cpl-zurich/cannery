"""Shared ingress oracle cache with fenced, resumable baseline attempts."""

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import JSON, CheckConstraint, DateTime, Engine, event, inspect, select
from sqlalchemy.engine import Connection
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Mapped, Mapper, Session, mapped_column

from cannery.artifacts import ArtifactRef
from cannery.database import Base
from cannery.git_intake import SnapshotBinding
from cannery.registry.registrations import aware


class BaselineState(Base):
    __tablename__ = "numerical_baselines"
    __table_args__ = (
        CheckConstraint("status IN ('queued','generating','ready','failed')"),
        CheckConstraint("attempts BETWEEN 0 AND 3"),
        CheckConstraint("(status = 'ready') = (artifact IS NOT NULL)"),
    )
    key: Mapped[str] = mapped_column(primary_key=True)
    spec_sha256: Mapped[str]
    snapshot: Mapped[dict[str, Any]] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(default="queued")
    attempts: Mapped[int] = mapped_column(default=0)
    attempt: Mapped[UUID | None]
    lease: Mapped[UUID | None]
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    deadline: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    due_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    artifact: Mapped[dict[str, Any] | None] = mapped_column(JSON(none_as_null=True))
    error: Mapped[str | None]


@event.listens_for(BaselineState, "before_update")
@event.listens_for(BaselineState, "before_delete")
def immutable_ready(mapper: Mapper[Any], connection: Connection, target: BaselineState) -> None:
    history = inspect(target).attrs.status.history
    if target.status == "ready" and not history.has_changes() or "ready" in history.deleted:
        raise ValueError("A completed CPU baseline is immutable")


def ensure_baseline(engine: Engine, key: str, spec_sha256: str, snapshot: SnapshotBinding) -> None:
    if any(re.fullmatch(r"[a-f0-9]{64}", value) is None for value in (key, spec_sha256)):
        raise ValueError("Invalid baseline identity")
    now = datetime.now(UTC)
    try:
        with Session(engine) as session, session.begin():
            if session.get(BaselineState, key) is None:
                session.add(
                    BaselineState(
                        key=key,
                        spec_sha256=spec_sha256,
                        snapshot=snapshot.model_dump(mode="json"),
                        deadline=now + timedelta(minutes=10),
                        due_at=now,
                    )
                )
    except IntegrityError:
        pass  # Another ingress may have inserted the same cache key.
    with Session(engine) as session:
        row = session.get(BaselineState, key)
        if row is None or row.spec_sha256 != spec_sha256:
            raise ValueError("Conflicting baseline identity")


@dataclass(frozen=True)
class BaselineClaim:
    key: str
    attempt: UUID
    token: UUID


def claim_baseline(engine: Engine, key: str) -> BaselineClaim | None:
    now = datetime.now(UTC)
    with Session(engine) as session, session.begin():
        row = session.scalar(
            select(BaselineState).where(BaselineState.key == key).with_for_update(skip_locked=True)
        )
        if row is None or row.status not in ("queued", "generating"):
            return None
        if row.lease_until is not None and aware(row.lease_until) > now:
            return None
        if aware(row.deadline) <= now:
            row.status, row.error = "failed", "deadline_exceeded"
            row.lease, row.lease_until = None, None
            return None
        if aware(row.due_at) > now:
            return None
        if row.status == "queued":
            if row.attempts >= 3:
                row.status, row.error = "failed", "attempts_exhausted"
                return None
            row.attempts += 1
            row.attempt = uuid4()
        assert row.attempt is not None
        row.status, row.lease = "generating", uuid4()
        row.lease_until = min(aware(row.deadline), now + timedelta(seconds=90))
        return BaselineClaim(key, row.attempt, row.lease)


def current_baseline(session: Session, claim: BaselineClaim) -> BaselineState | None:
    row = session.get(BaselineState, claim.key, with_for_update=True)
    now = datetime.now(UTC)
    if (
        row is None
        or row.status != "generating"
        or row.attempt != claim.attempt
        or row.lease != claim.token
        or row.lease_until is None
        or aware(row.lease_until) <= now
        or aware(row.deadline) <= now
    ):
        return None
    return row


def postpone_baseline(engine: Engine, claim: BaselineClaim) -> bool:
    with Session(engine) as session, session.begin():
        row = current_baseline(session, claim)
        if row is None:
            return False
        row.lease = None
        row.due_at = min(aware(row.deadline), datetime.now(UTC) + timedelta(seconds=5))
        row.lease_until = row.due_at
        return True


def finish_baseline(
    engine: Engine,
    claim: BaselineClaim,
    *,
    artifact: ArtifactRef | None = None,
    error: str | None = None,
    retryable: bool = False,
) -> bool:
    if (artifact is None) == (error is None):
        raise ValueError("Exactly one baseline artifact or error is required")
    if artifact is not None and (
        artifact.repository != "private" or artifact.revision != artifact.sha256
    ):
        raise ValueError("Baseline requires a pinned private artifact")
    with Session(engine) as session, session.begin():
        row = current_baseline(session, claim)
        if row is None:
            return False
        row.artifact = artifact.model_dump(mode="json") if artifact else None
        row.status = (
            "ready" if artifact else "queued" if retryable and row.attempts < 3 else "failed"
        )
        row.error, row.lease, row.lease_until = error, None, None
        if row.status == "queued":
            row.due_at = datetime.now(UTC) + timedelta(seconds=5 * row.attempts)
        return True
