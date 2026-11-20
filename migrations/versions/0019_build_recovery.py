"""Allow one operator-confirmed retry of failed device admission."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0019"
down_revision = "0018"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    schema = bind.scalar(sa.text("SELECT current_schema()"))
    if not isinstance(schema, str):
        raise RuntimeError("No schema for build recovery migration")
    search_path = f"pg_catalog, {bind.dialect.identifier_preparer.quote(schema)}, pg_temp"
    op.create_table(
        "build_recoveries",
        sa.Column("candidate_id", sa.Uuid(), sa.ForeignKey("candidates.id"), primary_key=True),
        sa.Column(
            "failed_attempt_id", sa.Uuid(), sa.ForeignKey("build_attempts.id"), nullable=False
        ),
        sa.Column("previous_build", JSONB(), nullable=False),
        sa.Column("log_sha256", sa.String(), nullable=False),
        sa.Column("recovered_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.execute("""CREATE TRIGGER immutable_build_recovery BEFORE UPDATE OR DELETE
        ON build_recoveries FOR EACH ROW EXECUTE FUNCTION reject_build_mutation()""")
    op.drop_constraint("submission_builds_attempts_check", "submission_builds", type_="check")
    op.create_check_constraint(
        "submission_builds_attempts_check", "submission_builds", "attempts BETWEEN 0 AND 4"
    )
    op.execute(f"""CREATE FUNCTION recover_failed_build(submission_id uuid, runtime_log text)
        RETURNS uuid LANGUAGE plpgsql SET search_path TO {search_path} AS $$
        DECLARE
          job submission_builds%ROWTYPE;
          recovered_at timestamptz;
        BEGIN
          SELECT * INTO STRICT job FROM submission_builds WHERE id = submission_id FOR UPDATE;
          IF job.status <> 'failed' OR job.error IS DISTINCT FROM 'runtime_failed'
            OR job.attempts <> 3 OR job.lease IS NOT NULL OR job.lease_until IS NOT NULL
            OR runtime_log IS NULL OR runtime_log !~ '^[0-9a-f]{{64}}$'
            OR NOT EXISTS (
              SELECT FROM build_attempts a JOIN build_results r USING (id)
              WHERE a.id = job.attempt_id AND a.candidate_id = job.id AND a.number = 3
                AND r.status = 'failed' AND r.error = 'runtime_failed'
                AND (
                  SELECT log.value->>'sha256'
                  FROM json_each(r.evidence->'logs') AS log
                  WHERE log.key ~ '^runtime-[0-9]+$'
                  ORDER BY length(log.key) DESC, log.key DESC LIMIT 1
                ) = runtime_log
            )
          THEN RAISE EXCEPTION 'Build has no matching recoverable runtime failure';
          END IF;
          recovered_at := clock_timestamp();
          INSERT INTO build_recoveries
          VALUES (job.id, job.attempt_id, to_jsonb(job), runtime_log, recovered_at);
          UPDATE submission_builds SET status = 'queued', error = NULL, due_at = recovered_at,
              deadline = recovered_at + interval '30 minutes' WHERE id = job.id;
          RETURN job.id;
        END $$""")
    op.execute("REVOKE EXECUTE ON FUNCTION recover_failed_build(uuid, text) FROM PUBLIC")


def downgrade() -> None:
    op.execute("LOCK TABLE submission_builds, build_recoveries IN ACCESS EXCLUSIVE MODE")
    op.execute("""DO $$ BEGIN
      IF EXISTS (SELECT FROM build_recoveries) THEN
        RAISE EXCEPTION 'Cannot downgrade with retained build recovery evidence';
      END IF;
    END $$""")
    op.execute("DROP FUNCTION recover_failed_build(uuid, text)")
    op.drop_table("build_recoveries")
    op.drop_constraint("submission_builds_attempts_check", "submission_builds", type_="check")
    op.create_check_constraint(
        "submission_builds_attempts_check", "submission_builds", "attempts BETWEEN 0 AND 3"
    )
