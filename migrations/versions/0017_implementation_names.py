"""Optional immutable implementation display names."""

import sqlalchemy as sa
from alembic import op

revision = "0017"
down_revision = "0016"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("submissions", sa.Column("implementation_name", sa.String(80), nullable=True))


def downgrade() -> None:
    op.drop_column("submissions", "implementation_name")
