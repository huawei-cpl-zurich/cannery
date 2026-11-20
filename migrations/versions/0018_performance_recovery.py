"""Allow an operator to recover expired timing without losing terminal evidence."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0018"
down_revision = "0017"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "performance_recoveries",
        sa.Column("job_key", sa.String(), sa.ForeignKey("performance_jobs.key"), primary_key=True),
        sa.Column("previous_attempts", sa.Integer(), nullable=False),
        sa.Column("previous_job", JSONB(), nullable=False),
        sa.Column("recovered_at", sa.DateTime(timezone=True), primary_key=True),
        sa.Column("new_deadline", sa.DateTime(timezone=True), nullable=False),
    )
    op.execute("""CREATE TRIGGER immutable_performance_recovery BEFORE UPDATE OR DELETE
        ON performance_recoveries FOR EACH ROW EXECUTE FUNCTION protect_performance_evidence()""")
    op.execute("""CREATE OR REPLACE FUNCTION protect_performance_evidence() RETURNS trigger
        LANGUAGE plpgsql SET search_path FROM CURRENT AS $$
        BEGIN
          IF TG_TABLE_NAME <> 'performance_jobs' THEN
            RAISE EXCEPTION 'Terminal performance evidence is immutable';
          END IF;
          IF OLD.status NOT IN ('ready', 'failed', 'blocked') THEN
            IF TG_OP = 'DELETE' THEN RETURN OLD; END IF;
            RETURN NEW;
          END IF;
          IF TG_OP = 'UPDATE' AND (
            (OLD.status = 'failed' AND OLD.error = 'deadline_exceeded')
            OR (
              OLD.kind = 'candidate' AND OLD.status = 'blocked'
              AND OLD.error = 'reference_timing_failed' AND EXISTS (
                SELECT FROM performance_jobs AS dependency
                JOIN performance_recoveries AS history ON history.job_key = dependency.key
                WHERE dependency.key = OLD.reference_key
                  AND dependency.status IN ('ready', 'queued', 'running')
                  AND history.previous_job->>'error' = 'deadline_exceeded'
              )
            )
          )
            AND OLD.attempts < 3 AND OLD.lease IS NULL AND OLD.lease_until IS NULL
            AND NEW.status = 'queued'
            AND NEW.error IS NOT DISTINCT FROM (
              SELECT 'awaiting_reference_timing'::varchar
              FROM performance_jobs AS pending
              WHERE OLD.kind = 'candidate' AND pending.key = OLD.reference_key
                AND pending.status IN ('queued', 'running')
            )
            AND (to_jsonb(OLD) - ARRAY['status', 'error', 'due_at', 'deadline'])
                = (to_jsonb(NEW) - ARRAY['status', 'error', 'due_at', 'deadline'])
            AND EXISTS (
              SELECT FROM performance_recoveries
              WHERE job_key = OLD.key AND previous_attempts = OLD.attempts
                AND previous_job = to_jsonb(OLD) AND recovered_at = NEW.due_at
                AND new_deadline = NEW.deadline
                AND (OLD.status = 'blocked' OR recovered_at >= OLD.deadline)
                AND new_deadline = recovered_at + interval '30 minutes'
            )
          THEN RETURN NEW;
          END IF;
          RAISE EXCEPTION 'Terminal performance evidence is immutable';
        END $$""")
    op.execute("""CREATE FUNCTION recover_expired_performance(submission_id uuid) RETURNS text
        LANGUAGE plpgsql AS $$
        DECLARE
          candidate performance_jobs%ROWTYPE;
          reference performance_jobs%ROWTYPE;
          queued_at timestamptz;
        BEGIN
          SELECT * INTO STRICT candidate FROM performance_jobs
          WHERE candidate_id = submission_id FOR UPDATE;
          SELECT * INTO STRICT reference FROM performance_jobs
          WHERE key = candidate.reference_key AND kind = 'reference' FOR UPDATE;
          queued_at := clock_timestamp();
          IF NOT coalesce(
            (candidate.status = 'failed' AND candidate.error = 'deadline_exceeded'
             AND candidate.deadline <= queued_at)
            OR (candidate.status = 'blocked' AND candidate.error = 'reference_timing_failed'),
            false
          ) OR candidate.attempts >= 3
            OR candidate.lease IS NOT NULL OR candidate.lease_until IS NOT NULL
          THEN RAISE EXCEPTION 'Candidate has no recoverable expired timing job';
          END IF;
          IF reference.status = 'failed' AND reference.error = 'deadline_exceeded'
            AND reference.deadline <= queued_at AND reference.attempts < 3
            AND reference.lease IS NULL AND reference.lease_until IS NULL
          THEN
            INSERT INTO performance_recoveries
            VALUES (reference.key, reference.attempts, to_jsonb(reference),
                    queued_at, queued_at + interval '30 minutes');
            UPDATE performance_jobs SET status = 'queued', error = NULL, due_at = queued_at,
                   deadline = queued_at + interval '30 minutes' WHERE key = reference.key;
          ELSIF reference.status NOT IN ('ready', 'queued', 'running')
            OR (reference.status IN ('queued', 'running') AND reference.deadline <= queued_at)
          THEN RAISE EXCEPTION 'Reference timing cannot be recovered';
          END IF;
          INSERT INTO performance_recoveries
          VALUES (candidate.key, candidate.attempts, to_jsonb(candidate),
                  queued_at, queued_at + interval '30 minutes');
          UPDATE performance_jobs SET status = 'queued', due_at = queued_at,
                 error = CASE WHEN reference.status <> 'ready' THEN 'awaiting_reference_timing' END,
                 deadline = queued_at + interval '30 minutes' WHERE key = candidate.key;
          RETURN candidate.key;
        END $$""")
    op.execute("REVOKE EXECUTE ON FUNCTION recover_expired_performance(uuid) FROM PUBLIC")


def downgrade() -> None:
    # A downgrade must not discard recovery history or weaken attempt provenance.
    op.execute("""DO $$ BEGIN
      IF EXISTS (SELECT FROM performance_recoveries) THEN
        RAISE EXCEPTION 'Cannot downgrade with retained timing recovery evidence';
      END IF;
    END $$""")
    op.execute("DROP FUNCTION recover_expired_performance(uuid)")
    op.drop_table("performance_recoveries")
    op.execute("""CREATE OR REPLACE FUNCTION protect_performance_evidence() RETURNS trigger
        LANGUAGE plpgsql AS $$ BEGIN
        IF TG_TABLE_NAME = 'performance_results' OR OLD.status IN ('ready','failed','blocked') THEN
          RAISE EXCEPTION 'Terminal performance evidence is immutable';
        END IF;
        IF TG_OP = 'DELETE' THEN RETURN OLD; END IF;
        RETURN NEW;
        END $$""")
