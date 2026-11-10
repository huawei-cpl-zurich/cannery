"""Durable submission intake leases and scratch ownership."""

import sqlalchemy as sa
from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "submission_intake",
        sa.Column("id", sa.Uuid(), sa.ForeignKey("submissions.id"), primary_key=True),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("deadline", sa.DateTime(timezone=True), nullable=False),
        sa.Column("due_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("lease", sa.Uuid()),
        sa.Column("lease_until", sa.DateTime(timezone=True)),
        sa.Column("scratch_token", sa.Uuid()),
        sa.Column("error", sa.String(64)),
        sa.CheckConstraint("status IN ('queued','intaking','awaiting_build','failed')"),
        sa.CheckConstraint("attempts >= 0"),
    )


def downgrade() -> None:
    op.drop_table("submission_intake")
