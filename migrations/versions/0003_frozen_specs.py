"""Immutable Git-backed specifications."""

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "kernels",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("namespace", sa.String(64), nullable=False),
        sa.Column("name", sa.String(64), nullable=False),
        sa.UniqueConstraint("namespace", "name"),
    )
    op.create_table(
        "spec_versions",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("kernel_id", sa.String(64), sa.ForeignKey("kernels.id"), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("content_sha256", sa.String(64), nullable=False),
        sa.Column("manifest", sa.JSON(), nullable=False),
        sa.Column("snapshot", sa.JSON(), nullable=False),
        sa.Column("file_sha256", sa.JSON(), nullable=False),
        sa.Column("smoke", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("kernel_id", "version"),
        sa.UniqueConstraint("kernel_id", "content_sha256"),
        sa.CheckConstraint("version > 0"),
    )
    op.execute("""CREATE FUNCTION reject_spec_mutation() RETURNS trigger LANGUAGE plpgsql AS $$
    BEGIN RAISE EXCEPTION 'Frozen spec versions cannot be updated or deleted'; END $$""")
    op.execute("""CREATE TRIGGER immutable_spec BEFORE UPDATE OR DELETE ON spec_versions
                  FOR EACH ROW EXECUTE FUNCTION reject_spec_mutation()""")


def downgrade() -> None:
    op.drop_table("spec_versions")
    op.execute("DROP FUNCTION reject_spec_mutation()")
    op.drop_table("kernels")
