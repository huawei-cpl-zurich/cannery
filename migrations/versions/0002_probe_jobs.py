"""Durable CPU probe jobs."""

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "probe_jobs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("effects", sa.Integer(), nullable=False),
        sa.CheckConstraint("status IN ('queued', 'complete', 'failed')"),
        sa.CheckConstraint("effects IN (0, 1)"),
    )


def downgrade() -> None:
    op.drop_table("probe_jobs")
