"""Durable numerical validation, separate from admitted binary state."""

import sqlalchemy as sa
from alembic import op

revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "submission_numerics",
        sa.Column("id", sa.Uuid(), sa.ForeignKey("candidates.id"), primary_key=True),
        sa.Column("baseline_key", sa.String(), sa.ForeignKey("numerical_baselines.key")),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("attempt_id", sa.Uuid()),
        sa.Column("lease", sa.Uuid()),
        sa.Column("lease_until", sa.DateTime(timezone=True)),
        sa.Column("deadline", sa.DateTime(timezone=True), nullable=False),
        sa.Column("due_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("error", sa.String()),
        sa.CheckConstraint("status IN ('queued','validating','passed','failed','blocked')"),
        sa.CheckConstraint("attempts BETWEEN 0 AND 3"),
    )
    op.create_table(
        "numerical_results",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("candidate_id", sa.Uuid(), sa.ForeignKey("candidates.id"), nullable=False),
        sa.Column("number", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("report", sa.JSON(none_as_null=True)),
        sa.Column("evidence", sa.JSON(), nullable=False),
        sa.Column("error", sa.String()),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("candidate_id", "number"),
        sa.CheckConstraint("status IN ('passed','failed')"),
    )
    op.execute("""CREATE FUNCTION reject_numerical_result_mutation() RETURNS trigger
    LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION 'Numerical results are immutable'; END $$""")
    op.execute("""CREATE TRIGGER immutable_numerical_result BEFORE UPDATE OR DELETE
    ON numerical_results FOR EACH ROW EXECUTE FUNCTION reject_numerical_result_mutation()""")


def downgrade() -> None:
    op.drop_table("numerical_results")
    op.drop_table("submission_numerics")
    op.execute("DROP FUNCTION reject_numerical_result_mutation()")
