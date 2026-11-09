# CANNery — Implementation Plan and Libraries

Status: engineering implementation notes | 2026-11-09

## 1. Implementation principles

Python across the platform; FastAPI for the HTTP API and HTML endpoints; prefer stable libraries. Keep CANNery's API independent of Hugging Face transport and the Ascend runtime. Build a thin capability-aware scheduler, not a new general-purpose task system. Treat any user-supplied Python, spec or compiler input as untrusted.

Work is divided into two repos:
- `cannery` hosts the services, CLI, benchmark workers, docs, schemas and UI.
- `cannery-kernels` tracks `huggingface/kernels` plus Ascend loader and compatibility support. Avoid inventing a parallel callable API if `get_kernel` can be extended. Implement Ascend build adapters in `cannery` first, to avoid a separate `kernel-builder` fork.

## 2. Library choices

| Area | Library | Notes |
|---|---|---|
| API | `fastapi`, `pydantic`, `uvicorn` | Explicit API versioning; generated OpenAPI |
| HTML | `jinja2`, `htmx` | Single server for API and UI |
| Styling | Tailwind CSS + daisyUI | Precompiled CSS in deployment; no Node server required |
| Charts | Apache ECharts | Cumulative best step line + language-colored scatter points |
| Tables | Tabulator | Sort/filter/export case-level µs tables |
| Admin | `sqladmin` + custom dashboards | Role-gated; do not expose worker/DB controls publicly |
| CLI | `typer`, `rich`, `httpx` | Typed commands and friendly job status |
| Persistence | `sqlalchemy` 2.x, `alembic`, `psycopg` | PostgreSQL authoritative |
| Queue | `dramatiq[redis]` | Redis-backed processing, retries; DB job state remains authoritative |
| Storage | `huggingface_hub` | Private repo snapshots, revisions, manifests |
| S3 alternative | `boto3` or `s3fs` | Optional later; keep backend contract generic |
| Benchmark | `torch`, `torch_npu`, `pytest`, `numpy` | Device timing and reference correctness |
| Metrics | `prometheus-client`, OpenTelemetry | Job rates, queue wait, worker health; admin only |
| Auth | Corporate OIDC with `authlib` | UI session + token-based CLI/worker access |
| Validation | `jsonschema` / Pydantic + YAML parser | Strict schemas; use safe YAML loading |
| Quality | `ruff`, `pytest`, `mypy` | CI on CPU without NPU |
| Packaging | `build`, `packaging` | Versioned wheels / runtime constraint checks |

License and dependency security review are required before adopting UI packages. Validate compatibility with the deployed corporate Python, torch_npu and CANN versions; the table is an initial selection, not a guarantee of device support.

## 3. Suggested package layout

```text
cannery/
  pyproject.toml
  src/cannery/
    api/                 # FastAPI routes, auth, dependencies
    web/                 # Jinja2 views, HTMX partials
    cli/                 # Typer commands
    models/              # SQLAlchemy + Pydantic contracts
    registry/            # specs, submissions, publications
    compatibility/       # device / language / ABI matching
    artifacts/           # HF and S3 backend interface
    scheduler/           # capability matching, per-device leases
    worker/              # daemon and heartbeat
    builder/             # Ascend C, Triton-Ascend, CATLASS adapters
    benchmark/           # correctness, device timing, profiles
    telemetry/           # status, ETA, audit logs
  templates/
  static/
  alembic/
  tests/
  docs/
    design.md
    implementation.md
    roadmap.md
    mockups/
```

## 4. Data model / invariants

Tables:
- `kernels`: id, namespace, name, owner, visibility, descriptive metadata.
- `spec_versions`: kernel_id, version, bundle_digest, bundle_uri, source provenance, suite_version, registered_by, immutable timestamps.
- `cases`: spec_version_id, case_id, metadata, required flag, deterministic generation contract.
- `submissions`: id, spec_version_id, author, device, language, source_digest, source_uri, git_url, git_commit_sha, repo_path, parent_submission_id, created_at.
- `artifacts`: submission_id, kind (source/binary/manifest), hash, URI, ABI/toolchain metadata.
- `jobs`: task_type, submission_id, status, queue timestamps, attempt, worker lease, failure category.
- `workers`: id, capabilities JSONB, runtime versions, last_heartbeat, status.
- `validation_runs`: submission_id, spec_digest, suite_revision, environment_digest, tolerance_profile_id, outcome, error statistics, case coverage.
- `benchmark_runs`: validation_run_id, device, environment cohort, per-case timings, baseline, geomean speedup, start/end times.
- `publications`: submission_id, actor, published_at, withdrawn_at, visibility.
- `tolerance_profiles`: immutable named `rtol`, `atol`, `equal_nan`, metric protocol and suite version.

Database constraints enforce unique `(kernel_id,spec_version)` and immutable digests. Use migration guards/triggers or API-level immutable writes plus audit logs; avoid update endpoints on immutable resources. Idempotency keys on registration and submissions prevent duplicates from retries.

## 5. API outline

```text
POST   /api/v1/kernels                     # register operator + spec
GET    /api/v1/kernels                     # global discovery
GET    /api/v1/kernels/{namespace}/{name}  # metadata
GET    /api/v1/kernels/{id}/specs/{v}      # frozen spec/manifest
POST   /api/v1/kernels/{id}/submissions    # upload or pinned Git submission
GET    /api/v1/submissions/mine             # all user jobs and statuses
GET    /api/v1/submissions/{id}            # private details and ETA
POST   /api/v1/submissions/{id}/publish    # authorized explicit publication
GET    /api/v1/leaderboard                 # device, spec, tolerance, case-suite filters
GET    /api/v1/leaderboard/history         # best-so-far records, one device
POST   /api/v1/benchmark-jobs              # run evaluation
GET    /api/v1/jobs/{id}                   # status, errors, expected completion time
POST   /api/v1/artifacts/resolve           # compatibility-aware source/binary selection
GET    /api/v1/admin/workers               # admin only
GET    /api/v1/admin/storage               # admin only
```

The API should never return provisional benchmark scores as final. The UI can show incremental stage progress, but reports become eligible for leaderboard only after complete correctness coverage, benchmark completion and explicit publication.

## 6. Registration and submission internals

**Registration path**
1. Authenticate and authorize namespace.
2. Validate YAML against strict `schema_version`.
3. Retrieve local upload or resolved Git commit + path; disallow moving branch/tag references as stored identity.
4. Validate PyTorch reference, case generators and shape/dtype contract in isolated service worker. Put an upper bound on spec dependency and fixture sizes.
5. Snapshot spec, canonicalize metadata, hash file manifest and environment lock; upload immutable bundle.
6. Store spec version record transactionally, with reference to uploaded content hash.

**Implementation path**
1. Resolve kernel + frozen spec; reject `reference.py` or spec changes in implementation payload.
2. Resolve pinned Git SHA or local snapshot; capture provenance and hashes.
3. Inspect device/language/toolchain metadata.
4. Enqueue compile → correctness → benchmark; each stage writes durable DB job transitions.
5. Publish measured error + runtime metadata privately. Allow explicit publication after policy checks.

For arbitrary user-submitted Python references, use separate sandbox workers with no internal network reachability and no production secrets. Avoid `pickle` or unrestricted imported modules in the main API process.

## 7. Benchmark engine and tolerance filtering

The reference and candidate run on the same case fixtures and declared environment, with deterministic seeds where possible. Device latency uses NPU-aware timing primitives and synchronization; ordinary Python `perf_counter` surrounding asynchronous launches is not sufficient. Warm up before repetition, exclude compilation and one-time initialization from steady-state kernel latency, and store compilation latency separately. Apply one benchmark per physical NPU lease to reduce contention. Capture clock/power/runtime environment when available.

Use per-case `torch.testing.assert_close` predicate:

```text
abs(actual - expected) <= atol + rtol * abs(expected)
```

Record additional error distributions and nonfinite counts. For the PoC, create a fixed, explicit set of tolerance profiles and validate them from original outputs; persist results by profile. A filter not already evaluated requires a new correctness run before an implementation can appear as eligible. It is **not safe** to infer arbitrary threshold validity from independent max absolute and relative error alone.

```python
import math
def geomean_speedup(cases):
    assert cases and all(c.baseline_us > 0 and c.kernel_us > 0 for c in cases)
    return math.exp(sum(math.log(c.baseline_us / c.kernel_us) for c in cases)
                    / len(cases))
```

Only use complete mandatory-case sets, matched device and baseline. Do not average microsecond latencies to compare kernels across differently shaped suites.

## 8. Scheduling and ETA

Worker registers hardware devices, CANN/torch_npu versions, language adapters, build/validate/benchmark operations, capacity and heartbeat. A scheduler matches capability predicates to pending jobs, takes a durable device lease (transactional database row / advisory lock), then issues work to Dramatiq. Heartbeat renewal and lease expiry handle crashed workers; execution is at-least-once so jobs and artifact writes must be idempotent.

Expose ETA to authors, not worker topology. Initial ETA estimate: queue position within compatible capacity + rolling per-stage runtime distributions for similar device, language and case count; return a range and confidence quality. Update after each job stage. No invented precision when estimates are sparse.

## 9. Runtime and distribution integration

Use `cannery-kernels` to fetch/load/call published kernels when possible, keeping `get_kernel` API compatibility. The CANNery server resolves artifacts by target device, language, CANN runtime, torch_npu, Python ABI and package constraints. Source artifacts remain downloadable independent of binary compatibility. `--no-detect` disables **local selection** but requires explicit variant choice; it never skips worker-side checks or signature/hash verification.

Build adapters in this repository handle `ascendc`, `triton` and `catlass`, producing source and optional binary artifacts with build manifests and reproducible toolchain provenance. Do not require `.run` as the CATLASS Python API; use the actual CATLASS callable convention.

## 10. Web implementation details

Global Best filter controls: kernel, device, spec, tolerance profile, optional dtype/case view. Results include individual `latency_us` values and geometric mean speedup. ECharts draws **one** cumulative-best step series for selected device/spec/suite/profile, plus colored point events for record-setting language, with tooltips linking to submission and Git SHA. A language switch changes filters, but it does not create a cross-device comparison.

My Submissions shows all author-owned revisions, statuses, logs, ETA and permission-gated publish controls. Running jobs show phase/status, never finalized benchmark claims. Admin navigation is role gated; workers, queue topology and storage never appear in the regular user navigation.

See [Design](design.md) and [Roadmap](roadmap.md).

## 11. Operational concerns / CI

- GitHub Actions: Python 3.11/3.12 lint/type/unit tests on CPU; avoid requiring CANN/NPU on hosted runners.
- Integration tests: PostgreSQL and Redis service containers; FakeArtifactStore and mock worker for reproducible jobs.
- Hardware tests: protected self-hosted A3/A5 runners or manually triggered internal benchmarks; never permit arbitrary PR code on privileged host runners.
- Threat model: untrusted Python and compilers, zip-slip, SSRF from Git URLs, artifact bombs, token leakage, cache poisoning, and abuse of benchmark resources.
- Configure time/memory/disk quotas, network isolation, max case count and artifact retention. Keep observability, backups and role-gated admin workflows.
