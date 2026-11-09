from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import JSON, DateTime, ForeignKey, String, UniqueConstraint, event
from sqlalchemy.engine import Connection
from sqlalchemy.orm import Mapped, Mapper, mapped_column

from cannery.database import Base


class SpecConflict(ValueError):
    pass


class Kernel(Base):
    __tablename__ = "kernels"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    namespace: Mapped[str] = mapped_column(String(64))
    name: Mapped[str] = mapped_column(String(64))
    __table_args__ = (UniqueConstraint("namespace", "name"),)


class SpecVersion(Base):
    __tablename__ = "spec_versions"
    __table_args__ = (
        UniqueConstraint("kernel_id", "version"),
        UniqueConstraint("kernel_id", "content_sha256"),
    )
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    kernel_id: Mapped[str] = mapped_column(ForeignKey("kernels.id"))
    version: Mapped[int]
    content_sha256: Mapped[str] = mapped_column(String(64))
    manifest: Mapped[dict[str, Any]] = mapped_column(JSON)
    snapshot: Mapped[dict[str, Any]] = mapped_column(JSON)
    file_sha256: Mapped[dict[str, str]] = mapped_column(JSON)
    smoke: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


@event.listens_for(SpecVersion, "before_update")
@event.listens_for(SpecVersion, "before_delete")
def immutable(mapper: Mapper[Any], connection: Connection, target: SpecVersion) -> None:
    raise SpecConflict("Frozen spec versions cannot be updated or deleted")
