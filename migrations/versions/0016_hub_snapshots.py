"""Immutable virtual Hub snapshot identities; artifact bytes stay in place."""

import sqlalchemy as sa
from alembic import op

revision = "0016"
down_revision = "0015"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "hub_snapshots",
        sa.Column("release_id", sa.Uuid(), sa.ForeignKey("releases.id"), primary_key=True),
        sa.Column("revision", sa.String(40), nullable=False, unique=True),
        sa.Column("descriptor", sa.JSON(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("hub_snapshots")
