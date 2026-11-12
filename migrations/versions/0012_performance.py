"""Retained reference and candidate performance jobs with immutable evidence."""

import sqlalchemy as sa
from alembic import op

revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "performance_jobs",
        sa.Column("key", sa.String(), primary_key=True),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("candidate_id", sa.Uuid(), sa.ForeignKey("candidates.id"), unique=True),
        sa.Column("reference_key", sa.String(), sa.ForeignKey("performance_jobs.key")),
        sa.Column("identity", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("attempt", sa.Uuid()),
        sa.Column("lease", sa.Uuid()),
        sa.Column("lease_until", sa.DateTime(timezone=True)),
        sa.Column("deadline", sa.DateTime(timezone=True), nullable=False),
        sa.Column("due_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("error", sa.String()),
        sa.CheckConstraint("kind IN ('reference','candidate')"),
        sa.CheckConstraint("status IN ('queued','running','ready','failed','blocked')"),
        sa.CheckConstraint("attempts BETWEEN 0 AND 3"),
        sa.CheckConstraint("(kind = 'candidate') = (candidate_id IS NOT NULL)"),
        sa.CheckConstraint("(kind = 'candidate') = (reference_key IS NOT NULL)"),
    )
    op.create_table(
        "performance_results",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("job_key", sa.String(), sa.ForeignKey("performance_jobs.key"), nullable=False),
        sa.Column("number", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("report", sa.JSON(none_as_null=True)),
        sa.Column("artifact", sa.JSON(none_as_null=True)),
        sa.Column("error", sa.String()),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("job_key", "number"),
        sa.CheckConstraint("status IN ('ready','failed','blocked')"),
        sa.CheckConstraint("status != 'ready' OR (artifact IS NOT NULL AND report IS NOT NULL)"),
    )
    op.execute("""CREATE FUNCTION protect_performance_evidence() RETURNS trigger
    LANGUAGE plpgsql AS $$ BEGIN
    IF TG_TABLE_NAME = 'performance_results' OR OLD.status IN ('ready','failed','blocked') THEN
      RAISE EXCEPTION 'Terminal performance evidence is immutable';
    END IF;
    IF TG_OP = 'DELETE' THEN RETURN OLD; END IF;
    RETURN NEW;
    END $$""")
    for table in ("performance_jobs", "performance_results"):
        op.execute(f"""CREATE TRIGGER immutable_performance BEFORE UPDATE OR DELETE
        ON {table} FOR EACH ROW EXECUTE FUNCTION protect_performance_evidence()""")


def downgrade() -> None:
    op.drop_table("performance_results")
    op.drop_table("performance_jobs")
    op.execute("DROP FUNCTION protect_performance_evidence()")
