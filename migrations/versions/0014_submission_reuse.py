"""Immutable unchanged-source links to completed executions."""

import sqlalchemy as sa
from alembic import op

revision = "0014"
down_revision = "0013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "submission_reuse",
        sa.Column("id", sa.Uuid(), sa.ForeignKey("candidates.id"), primary_key=True),
        sa.Column("original_id", sa.Uuid(), sa.ForeignKey("candidates.id"), nullable=False),
        sa.CheckConstraint("id <> original_id"),
    )
    op.create_index("ix_submission_reuse_original_id", "submission_reuse", ["original_id"])
    op.execute("""CREATE FUNCTION protect_submission_reuse() RETURNS trigger
    LANGUAGE plpgsql AS $$ BEGIN
      RAISE EXCEPTION 'Submission reuse is immutable';
    END $$""")
    op.execute("""CREATE TRIGGER immutable_submission_reuse
    BEFORE UPDATE OR DELETE ON submission_reuse FOR EACH ROW
    EXECUTE FUNCTION protect_submission_reuse()""")


def downgrade() -> None:
    op.drop_table("submission_reuse")
    op.execute("DROP FUNCTION protect_submission_reuse()")
