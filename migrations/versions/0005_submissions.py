"""Immutable implementation requests and retained candidates."""

import sqlalchemy as sa
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "submissions",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("namespace", sa.String(), nullable=False),
        sa.Column("spec_id", sa.Uuid(), sa.ForeignKey("spec_versions.id"), nullable=False),
        sa.Column("parent_id", sa.Uuid(), sa.ForeignKey("submissions.id")),
        sa.Column("source", sa.JSON(), nullable=False),
        sa.Column("target", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("parent_id <> id"),
    )
    op.create_index("ix_submissions_owner_time", "submissions", ["namespace", "created_at"])
    op.create_table(
        "candidates",
        sa.Column("id", sa.Uuid(), sa.ForeignKey("submissions.id"), primary_key=True),
        sa.Column("binding_id", sa.Uuid(), sa.ForeignKey("submissions.id"), nullable=False),
        sa.Column("binding", sa.JSON(), nullable=False),
        sa.Column("manifest", sa.JSON(), nullable=False),
        sa.Column("snapshot", sa.JSON(), nullable=False),
        sa.Column("content_sha256", sa.String(), nullable=False),
        sa.Column("file_sha256", sa.JSON(), nullable=False),
    )
    op.execute("""CREATE FUNCTION reject_submission_mutation() RETURNS trigger
    LANGUAGE plpgsql AS $$
    BEGIN RAISE EXCEPTION 'Submission requests and accepted candidates are immutable'; END $$""")
    for table in ("submissions", "candidates"):
        op.execute(f"""CREATE TRIGGER immutable_{table} BEFORE UPDATE OR DELETE ON {table}
                       FOR EACH ROW EXECUTE FUNCTION reject_submission_mutation()""")


def downgrade() -> None:
    op.drop_table("candidates")
    op.drop_table("submissions")
    op.execute("DROP FUNCTION reject_submission_mutation()")
