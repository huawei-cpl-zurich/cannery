"""Internal CPU probe; no submitted code is executed."""

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy import CheckConstraint, DateTime, Engine, String, insert, select, update
from sqlalchemy.orm import Mapped, mapped_column

from cannery.database import Base


class ProbeJob(Base):
    __tablename__ = "probe_jobs"
    __table_args__ = (
        CheckConstraint("status IN ('queued', 'complete', 'failed')"),
        CheckConstraint("effects IN (0, 1)"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(16), default="queued")
    effects: Mapped[int] = mapped_column(default=0)


def create_probe(engine: Engine) -> UUID:
    job_id = uuid4()
    with engine.begin() as connection:
        connection.execute(
            insert(ProbeJob).values(
                id=job_id,
                created_at=datetime.now(UTC),
                status="queued",
                effects=0,
            )
        )
    return job_id


def complete_probe(engine: Engine, job_id: str) -> None:
    """A single atomic transaction makes concurrent/repeated deliveries harmless."""
    key = UUID(job_id)
    with engine.begin() as connection:
        connection.execute(
            update(ProbeJob)
            .where(ProbeJob.id == key, ProbeJob.status == "queued")
            .values(status="complete", effects=1)
        )


def fail_probe(engine: Engine, job_id: str) -> None:
    with engine.begin() as connection:
        connection.execute(
            update(ProbeJob)
            .where(ProbeJob.id == UUID(job_id), ProbeJob.status == "queued")
            .values(status="failed")
        )


def pending_probes(engine: Engine) -> list[str]:
    with engine.connect() as connection:
        return [
            str(key)
            for key in connection.scalars(select(ProbeJob.id).where(ProbeJob.status == "queued"))
        ]


def reconcile_probes(engine: Engine) -> list[str]:
    """Expire old probes and return a bounded replay batch after lost dispatch."""
    cutoff = datetime.now(UTC) - timedelta(seconds=60)
    with engine.begin() as connection:
        connection.execute(
            update(ProbeJob)
            .where(
                ProbeJob.status == "queued",
                ProbeJob.created_at < cutoff,
            )
            .values(status="failed")
        )
        return [
            str(key)
            for key in connection.scalars(
                select(ProbeJob.id)
                .where(ProbeJob.status == "queued")
                .order_by(ProbeJob.created_at)
                .limit(100)
            )
        ]
