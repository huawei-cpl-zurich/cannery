"""Durable registration jobs and fenced attempts."""

import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "registrations",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("namespace", sa.String(64), nullable=False),
        sa.Column("source", sa.JSON(), nullable=False),
        sa.Column("expected_name", sa.String(64)),
        sa.Column("allow_new_version", sa.Boolean(), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("deadline", sa.DateTime(timezone=True), nullable=False),
        sa.Column("due_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("lease", sa.Uuid()),
        sa.Column("lease_until", sa.DateTime(timezone=True)),
        sa.Column("error", sa.String(32)),
        sa.Column("cleanup_handle", sa.String(96)),
        sa.Column("spec_id", sa.Uuid(), sa.ForeignKey("spec_versions.id")),
        sa.CheckConstraint("status IN ('queued','running','complete','failed')"),
        sa.CheckConstraint("attempts BETWEEN 0 AND 3"),
        sa.CheckConstraint("(status = 'complete') = (spec_id IS NOT NULL)"),
    )
    op.create_index("ix_registrations_status", "registrations", ["status", "created_at"])


def downgrade() -> None:
    op.drop_table("registrations")
