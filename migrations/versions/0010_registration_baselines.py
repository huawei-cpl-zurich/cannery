"""Resume prepared numerical ingress without repeating Git intake."""

import sqlalchemy as sa
from alembic import op

revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("registrations", sa.Column("prepared_snapshot", sa.JSON(none_as_null=True)))
    op.add_column("registrations", sa.Column("baseline_key", sa.String()))
    op.create_foreign_key(
        "registration_baseline", "registrations", "numerical_baselines", ["baseline_key"], ["key"]
    )
    op.create_check_constraint(
        "prepared_baseline_pair",
        "registrations",
        "(prepared_snapshot IS NULL) = (baseline_key IS NULL)",
    )


def downgrade() -> None:
    op.drop_constraint("prepared_baseline_pair", "registrations", type_="check")
    op.drop_constraint("registration_baseline", "registrations", type_="foreignkey")
    op.drop_column("registrations", "baseline_key")
    op.drop_column("registrations", "prepared_snapshot")
