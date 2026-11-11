"""Shared ingress CPU oracle cache, frozen after successful publication."""

import sqlalchemy as sa
from alembic import op

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "numerical_baselines",
        sa.Column("key", sa.String(), primary_key=True),
        sa.Column("spec_sha256", sa.String(), nullable=False),
        sa.Column("snapshot", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("attempt", sa.Uuid()),
        sa.Column("lease", sa.Uuid()),
        sa.Column("lease_until", sa.DateTime(timezone=True)),
        sa.Column("deadline", sa.DateTime(timezone=True), nullable=False),
        sa.Column("due_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("artifact", sa.JSON(none_as_null=True)),
        sa.Column("error", sa.String()),
        sa.CheckConstraint("status IN ('queued','generating','ready','failed')"),
        sa.CheckConstraint("attempts BETWEEN 0 AND 3"),
        sa.CheckConstraint("(status = 'ready') = (artifact IS NOT NULL)"),
    )
    op.execute("""CREATE FUNCTION reject_ready_baseline_mutation() RETURNS trigger
    LANGUAGE plpgsql AS $$ BEGIN
    IF OLD.status = 'ready' THEN RAISE EXCEPTION 'Completed CPU baseline is immutable'; END IF;
    IF TG_OP = 'DELETE' THEN RETURN OLD; END IF;
    RETURN NEW; END $$""")
    op.execute("""CREATE TRIGGER immutable_ready_baseline BEFORE UPDATE OR DELETE
    ON numerical_baselines FOR EACH ROW EXECUTE FUNCTION reject_ready_baseline_mutation()""")


def downgrade() -> None:
    op.drop_table("numerical_baselines")
    op.execute("DROP FUNCTION reject_ready_baseline_mutation()")
