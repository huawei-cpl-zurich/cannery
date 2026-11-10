"""Durable binary admission attempts and immutable outcomes."""

import sqlalchemy as sa
from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "build_attempts",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("candidate_id", sa.Uuid(), sa.ForeignKey("candidates.id"), nullable=False),
        sa.Column("number", sa.Integer(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("candidate_id", "number"),
    )
    op.create_table(
        "build_results",
        sa.Column("id", sa.Uuid(), sa.ForeignKey("build_attempts.id"), primary_key=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("artifact", sa.JSON(), nullable=True),
        sa.Column("error", sa.String()),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("status IN ('complete','failed')"),
    )
    op.create_table(
        "submission_builds",
        sa.Column("id", sa.Uuid(), sa.ForeignKey("candidates.id"), primary_key=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("attempt_id", sa.Uuid(), sa.ForeignKey("build_attempts.id")),
        sa.Column("deadline", sa.DateTime(timezone=True), nullable=False),
        sa.Column("due_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("lease", sa.Uuid()),
        sa.Column("lease_until", sa.DateTime(timezone=True)),
        sa.Column("error", sa.String()),
        sa.CheckConstraint("status IN ('queued','building','awaiting_validation','failed')"),
        sa.CheckConstraint("attempts >= 0 AND attempts <= 3"),
    )
    op.execute("""CREATE FUNCTION reject_build_mutation() RETURNS trigger
    LANGUAGE plpgsql AS $$
    BEGIN RAISE EXCEPTION 'Build attempts and results are immutable'; END $$""")
    for table in ("build_attempts", "build_results"):
        op.execute(f"""CREATE TRIGGER immutable_{table} BEFORE UPDATE OR DELETE ON {table}
                       FOR EACH ROW EXECUTE FUNCTION reject_build_mutation()""")


def downgrade() -> None:
    for table in ("submission_builds", "build_results", "build_attempts"):
        op.drop_table(table)
    op.execute("DROP FUNCTION reject_build_mutation()")
