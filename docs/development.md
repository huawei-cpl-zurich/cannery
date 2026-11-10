# Local development and CPU acceptance

M0 supplies the CPU control-plane foundation; M1 adds Git-backed immutable
specification registration and isolated CPU reference diagnostics. M2 adds Git-only
source submissions and commit-only revisions. NPU builds, benchmarks and publication
remain later milestones.
No source-upload endpoint exists. No A3/A5 access is needed for these checks.

## Install and check

Use Python 3.12 or 3.13:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
ruff check .
ruff format --check .
mypy
pytest
python -m build
```

## Run the local stack

Requires Docker with Compose. Development database credentials in Compose are
local-only; PostgreSQL and Redis have no published host ports. The API binds only
to loopback. Reserve persistent volumes for PostgreSQL and per-kernel Git snapshots.

```bash
docker compose up --build -d --wait --scale worker=2
cannery health --url http://127.0.0.1:8000
```

Compose completes Alembic migrations before starting the API, workers and probe
reconciler. `/health/live` tests process responsiveness; `/health/ready` requires
PostgreSQL and Redis. The API never executes submitted code.

The web shell is at http://127.0.0.1:8000. By default requests are anonymous and
private/admin pages deny access. To inspect user navigation locally, start with
`CANNERY_DEV_IDENTITY_ENABLED=true docker compose up --build -d --wait`.
Development identity is fixed server-side, has the user role, and is not login.
For a local admin identity set `CANNERY_DEV_ROLE=admin` as well;
never enable development identity in a shared deployment. Real OIDC is deferred.

The internal CPU probe has a 60-second lifetime, three actor retries, and a failure
callback. The reconciler replays queued IDs every five seconds in batches of 100
and marks expired jobs failed after database recovery. Duplicate deliveries are
safe. This is foundation verification, not a device scheduler or arbitrary job API.

## One-command acceptance

```bash
python scripts/compose_acceptance.py
```

This creates a separate task-named Compose project on a temporary host port and
cleans its containers, volumes and generated images afterward. It tests migrations,
web/CLI, fake artifacts, two workers, duplicate deliveries, lost dispatch, Redis
outage and PostgreSQL pause/recovery. Ordinary `docker compose down` preserves
local data; add `--volumes` only when intentionally discarding that project's data.
CPU CI runs this script, CPU reference/registration/submission acceptance, and Python
3.12/3.13 lint, typing, tests and wheel installation checks. Tests also simulate HF retry delays, cache contention and
corruption without live credentials, using a disposable SDK process with a deadline.

## Artifact configuration

The default `CANNERY_ARTIFACT_BACKEND=fake` stores test bytes under
`CANNERY_ARTIFACT_ROOT` (default `/tmp/cannery-artifacts`). The fake is not HF
publication evidence and should not contain valuable persistent artifacts.

For real SDK use outside Compose, set `CANNERY_ARTIFACT_BACKEND=hf`,
`CANNERY_HF_REPOSITORY=org/private-repo`, and `CANNERY_HF_TOKEN` in the process
environment. Do not commit credentials. The existing model repository must be
private. The adapter uses immutable revisions and verifies SHA-256 before returning
bytes or an uploaded reference. Each operation has a 30-second subprocess deadline;
Xet is disabled inside that process so it can be terminated cleanly. A timed-out
upload may leave an unreferenced remote artifact; no publication is recorded by M0.
Live HF integration remains unverified until a separately authorized credentialed test.

Git intake implements operator-scoped HTTPS access, safe selected-file export,
synthetic snapshot lineage and upstream/internal SHA mapping. Protected internal
refs retain selected files after upstream deletion and garbage collection, without
retaining unrelated upstream history or relying on a temporary checkout.

## M1 registration

See [registration](registration.md) for the strict manifest/environment lock, operator
HTTPS repository rules, local development identity, CLI/API and web flow.
Run `python scripts/registration_acceptance.py` for private Git, two registrars,
actual CPU PyTorch diagnostics, immutable versions and service/lease recovery.
The test-only Compose override contains synthetic credentials and cleans its own
project resources. NPU validation is outside M1.

## M2 submissions

See [submissions and revisions](submissions.md) for strict `submission.yaml`,
CLI/API/web usage and the inherited source binding.
`python scripts/submission_acceptance.py` runs the actual user journey with two
registrars, private Git, outages, stale leases and retained-source recovery.
Intake success is `awaiting_build` with unknown ETA; no candidate code executes.
Rebuild local services with your existing operator environment to use merged changes.
