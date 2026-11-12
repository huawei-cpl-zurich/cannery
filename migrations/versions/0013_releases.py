"""Pinned releases and durable publication intent, jobs and audit history."""

import sqlalchemy as sa
from alembic import op

revision = "0013"
down_revision = "0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "releases",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "candidate_id", sa.Uuid(), sa.ForeignKey("candidates.id"), nullable=False, unique=True
        ),
        sa.Column("spec_id", sa.Uuid(), sa.ForeignKey("spec_versions.id"), nullable=False),
        sa.Column("device", sa.String(), nullable=False),
        sa.Column("language", sa.String(), nullable=False),
        sa.Column("cohort_sha256", sa.String(), nullable=False),
        sa.Column("runtime_sha256", sa.String(), nullable=False),
        sa.Column("score", sa.Float()),
        sa.Column("manifest", sa.JSON(), nullable=False),
        sa.Column("document", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("candidate_id", "id"),
    )
    for name in ("spec_id", "cohort_sha256", "runtime_sha256"):
        op.create_index(f"ix_releases_{name}", "releases", [name])
    op.create_table(
        "release_tolerances",
        sa.Column("release_id", sa.Uuid(), sa.ForeignKey("releases.id"), primary_key=True),
        sa.Column("rtol", sa.Float(), primary_key=True),
        sa.Column("atol", sa.Float(), primary_key=True),
    )
    op.create_table(
        "publications",
        sa.ForeignKeyConstraint(
            ["candidate_id", "release_id"], ["releases.candidate_id", "releases.id"]
        ),
        sa.Column("candidate_id", sa.Uuid(), sa.ForeignKey("candidates.id"), primary_key=True),
        sa.Column("release_id", sa.Uuid(), unique=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("actor", sa.JSON(), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint("status IN ('publishing','published','withdrawn','failed')"),
        sa.CheckConstraint("generation >= 1"),
        sa.CheckConstraint(
            "status != 'published' OR (release_id IS NOT NULL AND published_at IS NOT NULL)"
        ),
    )
    op.create_table(
        "publication_jobs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "candidate_id", sa.Uuid(), sa.ForeignKey("publications.candidate_id"), nullable=False
        ),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("lease", sa.Uuid()),
        sa.Column("lease_until", sa.DateTime(timezone=True)),
        sa.Column("deadline", sa.DateTime(timezone=True), nullable=False),
        sa.Column("due_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("error", sa.String()),
        sa.UniqueConstraint("candidate_id", "generation"),
        sa.CheckConstraint("status IN ('queued','running','complete','failed','cancelled')"),
        sa.CheckConstraint("attempts BETWEEN 0 AND 3"),
    )
    op.create_table(
        "publication_events",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "candidate_id", sa.Uuid(), sa.ForeignKey("publications.candidate_id"), nullable=False
        ),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("actor", sa.JSON(), nullable=False),
        sa.Column("details", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("kind IN ('requested','published','withdrawn','retry','failed')"),
    )
    op.execute("""CREATE FUNCTION protect_release_evidence() RETURNS trigger
    LANGUAGE plpgsql AS $$ BEGIN
    IF TG_TABLE_NAME != 'publication_jobs' THEN
      RAISE EXCEPTION 'Release evidence is immutable';
    END IF;
    IF OLD.status IN ('complete','failed','cancelled') THEN
      RAISE EXCEPTION 'Terminal publication jobs are immutable';
    END IF;
    IF TG_OP = 'DELETE' THEN RETURN OLD; END IF;
    RETURN NEW;
    END $$""")
    for table in ("releases", "release_tolerances", "publication_events", "publication_jobs"):
        op.execute(f"""CREATE TRIGGER immutable_release BEFORE UPDATE OR DELETE ON {table}
        FOR EACH ROW EXECUTE FUNCTION protect_release_evidence()""")


def downgrade() -> None:
    for table in (
        "publication_events",
        "publication_jobs",
        "publications",
        "release_tolerances",
        "releases",
    ):
        op.drop_table(table)
    op.execute("DROP FUNCTION protect_release_evidence()")
