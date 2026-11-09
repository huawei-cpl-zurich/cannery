# CANNery — Implementation and Library Plan

Status: revised PoC implementation contract · 2026-11-09

See [Architecture](architecture.md) for services, data movement and storage; [User Story](user-story.md) for UI, CLI, REST and Hugging Face consumption; [Design](design.md) for product requirements; [Roadmap](roadmap.md) for milestones.

## 1. Scope and boundaries

CANNery accepts **only Git-backed specification and implementation code**. Registration requires an external Git remote, full SHA and path; implementation revisions can supply only a new commit using a saved source binding. The server imports accepted source subtrees into **per-kernel Git snapshot repositories**. It does not ingest local source archives or upload source bundles to Hugging Face.

An implementation must declare a **reproducible build plan**. A capability-matched worker builds binary candidates during admission, then runs correctness against the frozen PyTorch spec and benchmarks. Generated binaries, manifests and detailed logs land in **private HF repositories**; metadata, Git identities, scores, states and artifact pointers land in PostgreSQL.

Initial supported adapters: **Ascend C** and **CATLASS DSL**. Future adapters: **Triton-Ascend** and **PyPTO**. Device tracks: A3 and A5. The runtime integration lives in `cannery-kernels`, the Hugging Face kernels fork.

## 2. Chosen libraries

| Subsystem | Python / frontend libraries | Notes |
|---|---|---|
| API / website | FastAPI, Pydantic, Uvicorn, Jinja2, HTMX | One deployable FastAPI process initially |
| UI controls | Tailwind CSS + daisyUI, Tabulator, Apache ECharts | Charts: single best-so-far step series per device |
| CLI | Typer, Rich, httpx | Git-only commands, inspection, build/validate/report status |
| Database | PostgreSQL, SQLAlchemy 2, Alembic, psycopg | Canonical metadata, immutable states, publication |
| Dispatch | Redis + Dramatiq | Jobs carry IDs only, PostgreSQL authoritative |
| Git operations | system `git` via controlled subprocess | Restricted temporary Git fetch, selected-path import, internal snapshot commit, full-SHA verification, restricted URLs; prefer Git CLI over reimplementing Git |
| Git credentials | corporate secret manager / deploy token | Referenced by ID; never place secrets in manifests |
| HF binary store | `huggingface_hub` | Pinned revisions, private repositories, checksum verification |
| Alternative binary store | boto3 (later) | Storage abstraction; preserve HF-compatible publication path |
| Device adapter | CANN/Ascend C, CATLASS compiler API, torch_npu | Target-specific; validate actual versioned build APIs |
| Testing | pytest, PyTorch `torch.testing.assert_close`, NumPy | Numerical correctness and device timing |
| Identity | corporate OIDC via Authlib | Separate user/admin/worker roles |
| Telemetry | prometheus-client, OpenTelemetry | Job status, ETA, device health |
| Quality | ruff, pytest, mypy | CPU CI, isolated hardware integration tests |

Hugging Face's kernel metadata recognizes `cann` as a backend, but its builder does not provide our Ascend build pipeline. CANNery owns the two adapters. See https://huggingface.co/docs/kernels/kernel-requirements.

## 3. Modules and services

```text
src/cannery/
  api/                   FastAPI REST, user/admin routes
  web/                   Jinja2 + HTMX
  cli/                   Typer
  registry/              kernel, immutable spec, submission, publication
  git_intake/            external Git fetch, selected-path snapshot commits, provenance
  build/                 adapters/ascendc.py and adapters/catlass.py
  benchmark/             reference correctness, timing, tolerance profiles
  scheduler/             worker capability matching and device leases
  worker/                trusted coordinator + sandbox executor
  artifacts/             HF binary and run evidence store
  compatibility/         device, CANN/runtime and ABI matching
  models/                SQLAlchemy + Pydantic schemas
  telemetry/             logs, progress, ETA, audit
```

Deployment units: reverse proxy, FastAPI web/API, scheduler, Redis, PostgreSQL, Git intake / Git snapshot repositories, A3 worker, A5 worker. Per-kernel Git snapshot repositories use a small backed-up persistent volume. Workers use isolated execution containers; the main API process never imports arbitrary submitted Python.

## 4. Data model

- `kernels`: namespace/name, owner, descriptive attributes.
- `git_repositories`: canonical remote, credential secret reference, tenant access and allowed scope.
- `git_commits`: repo, full SHA, internal snapshot commit SHA, upstream SHA, source path, original tree hash and imported content digest.
- `spec_versions`: immutable spec version, Git repo/commit/path, frozen spec file digests, suite/environment lock digest.
- `source_bindings`: repo, path, entrypoint, immutable spec ID, device, language and build adapter.
- `submissions`: binding, commit, parent submission, author, source tree digest, admission status.
- `build_plans`: normalized adapter/version/options/toolchain target; frozen per submission.
- `build_runs`: submission, worker/toolchain fingerprint, status, compilation log and binary digest/URI.
- `artifacts`: binary kind, HF repo/revision, checksum, ABI constraints, specialization coverage.
- `validation_runs`: spec/case/profile version, per-case outcome/errors, completed/failed.
- `benchmark_runs`: device, environment cohort, per-case µs, baseline, geomean, trace/evidence URI.
- `publications`: submission, pinned HF release mapping, publisher, visibility and timestamps.
- `jobs`/`workers`/`device_leases`: durable execution state, capabilities, capacity and failure recovery.

Use DB constraints to enforce immutability, a unique spec version per kernel, and immutable source binding/commit on submissions. Treat Git source tree hashes as useful verification, supplementary to retaining the imported selected files in a Git snapshot commit.

## 5. Git-only registration and submission

```text
POST /api/v1/kernels
  {namespace,name,source:{type:"git",url,commit,path}}

POST /api/v1/kernels/{namespace}/{name}/submissions
  {spec_version,source:{type:"git",url,commit,path},device,language}

POST /api/v1/submissions/{id}/revisions
  {commit:"<full SHA>"}      # inherits repo/path/spec/device/lang/build binding

GET /api/v1/submissions/mine
GET /api/v1/jobs/{job_id}
POST /api/v1/submissions/{id}/publish
GET /api/v1/leaderboard?device=a5&tolerance_profile=...
GET /api/v1/leaderboard/history?device=a5&...
POST /api/v1/artifacts/resolve
GET /api/v1/admin/workers
GET /api/v1/admin/git-snapshots
GET /api/v1/admin/storage
```

No upload-init or local archive paths. CLI: `cannery register --repo ... --commit ... --path ...`, `cannery submit ... --repo ... --commit ... --path ...`, `cannery revise <id> --commit <SHA>`.

Git intake: authorize remote and namespace, validate full SHA and path, fetch the commit with resource/time/network policy, copy only selected Git-tree files plus declared dependencies into a per-kernel internal Git repo as a new immutable snapshot commit, store upstream-to-internal SHA mapping, and check out only that snapshot in the sandbox, read strict YAML manifests and hash files. No arbitrary Git hooks or privileged build scripts. Reject admission if the commit cannot be mirrored.

A changed PyTorch spec YAML/reference/cases creates `spec-v2`; it does not mutate an accepted `spec-v1`. An implementation-only SHA change creates a new submission inheriting frozen fields. Optionally deduplicate identical built binary hashes while retaining separate submission records.

## 6. Build contract and adapters

The checked-in implementation `submission.yaml` declares source files, entrypoint, target and an allowlisted adapter with typed options. Example:

```yaml
schema_version: 1
kernel: team/fa_xxx
spec_version: 1
implementation:
  device: a5
  language: catlass
  entrypoint: kernel.py:forward
  build:
    adapter: catlass
    inputs: [kernel.py]
    artifact: fa_xxx_a5
    options:
      optimization: release
validation:
  requested_rtol: 0.001
  requested_atol: 0.0001
```

The trusted wrapper validates inputs, toolchain versions, supported specialization set and output paths. Its interface is:

```python
class BuildAdapter(Protocol):
    def preflight(self, git_tree: Path, manifest: SubmissionManifest, env: Toolchain) -> BuildPlan: ...
    def build(self, plan: BuildPlan, output_dir: Path) -> BuildResult: ...
    def package(self, result: BuildResult, target: RuntimeTarget) -> PackageManifest: ...
```

`BuildResult` includes generated binary files, exported call contract, toolchain fingerprint, device, supported shapes/dtypes, ABI dependencies, logs and hashes. The worker controls compiler invocations and sandbox boundaries. It must not execute unrestricted `build.sh` with host privileges.

- **Ascend C adapter (PoC):** compile/link CANN source to agreed loadable binary/library; pin CANN headers/compiler flags; record ABI and host wrapper. Use real version-specific supported toolchain.
- **CATLASS adapter (PoC):** use CATLASS' supported Pythonic compilation/JIT API to perform ahead-of-publication compilation for explicit shape/dtype specializations, export its actual device binary and any required loader glue. Research export capabilities early; **reject** unsupported specialization modes rather than calling a source-only JIT artifact a binary.
- **Triton-Ascend (later):** compile and package relevant target variants with equivalent metadata.
- **PyPTO (later):** PyPTO compiler/runtime adapter, with its separately pinned toolchain/runtime dependencies, not PTO-DSL.

Build is a mandatory admission/validation stage. A failed build preserves logs and private state. The generated binary is staged in HF and can be published only after correctness and full benchmark completion; keep an immutable build manifest. Never accept a prebuilt author binary as a substitute for source/build validation.

## 7. Validation, benchmark and leaderboard

Every kernel spec is an author-provided, frozen **PyTorch reference** plus case generator. Candidate binary and reference are evaluated in the same compatible environment. Correctness follows explicit `atol + rtol * abs(reference)`, with independent measured errors and a finite versioned set of tolerance profiles; unknown tolerance pairs require revalidation. Store per-case evidence.

Benchmark only passing candidates. Record synchronized device timing, warmup, median/p95 microseconds and baseline microseconds for each required case. `geomean_speedup = exp(mean(log(baseline_us/kernel_us)))`. Match device, suite, tolerance, and environment; do not compare A3 and A5. A completed and explicitly published result can enter Global Best. Display best-over-time as a single cumulative step curve with language-colored new-record markers.

The user only sees validation stage, logs and empirical ETA (or insufficient data), not worker identities or storage architecture. Admin-only pages handle workers, Git snapshots, queue and storage.

## 8. Artifact publication and Hugging Face

Published binaries are uploaded to private Hugging Face repositories with an immutable revision and a `cannery-kernels`-compatible loader manifest, source Git SHA/digest, toolchain fingerprint, ABI constraints, target device, language and specialization coverage. HF is not the source store. A source link resolves to the original Git URL and full SHA (or authorized internal mirror) instead.

Upload and verify HF files *before* making the PostgreSQL release globally discoverable; reconcile abandoned staging artifacts on failure. Private HF publication does not automatically permit external access. Local compatible binaries are selected by environment detection; `--no-detect` requires explicit variant fields. The extended Hugging Face runtime uses a pinned repo/revision and backend `cann`, with an explicit remote-code trust policy.

## 9. Testing and operational concerns

CPU CI tests Git intake with a local upstream Git fixture, per-kernel snapshot commits and protected refs and commit-only revisions, YAML validation, adapter preflight, fake binary artifacts, PostgreSQL/Redis/MockHF integration and fault recovery. Hardware tests use protected internal runners on A3 and A5 and verify real build/export, correctness and timing. Run untrusted code only in isolated NPU workers with resource/network limits. Git snapshot repositories and PG must be backed up; HF generated binaries are independently versioned. Use Prometheus/OpenTelemetry for scheduler/builder telemetry, with admin access.

## 10. PoC engineering risk

The **CATLASS binary export contract** is the biggest early technical unknown: whether every supported specialization can be compiled and redistributed in a loader-compatible form. Prioritize a proof of compilation→artifact→load→execute before implementing broad language coverage. Ascend C package ABI and torch_npu interop are the second critical test. Both need explicit A3/A5 validation with pinned toolchains.

### Subtree importer implementation contract

Use a transient constrained Git fetch to resolve the external full SHA, then safely export selected paths into a clean **per-kernel internal Git repository** as a new commit. Store both upstream and internal commit SHAs in PostgreSQL, with declared dependency paths and content digests. The internal commit lineage represents *accepted CANNery revisions*, not full upstream directory history. The build worker checks out the internal snapshot; admission never executes Git hooks. Git blob deduplication bounds incremental storage, but fetched upstream objects may temporarily occupy more disk. See [Architecture](architecture.md#git-subtree-snapshot-policy).
