# CANNery — PoC Roadmap

Status: proposed vertical slices | 2026-11-09

The roadmap prioritizes a **working user journey at every milestone**. Each slice includes API, CLI, minimal UI, storage, and tests. Hardware availability and runtime integration are the largest uncertainties; durations are estimates for planning, not commitments.

## Summary

| Milestone | User-visible outcome | Critical dependencies | Indicative effort |
|---|---|---|---|
| 0. Foundation | Local server, CLI, website shell, CI | Existing repositories | 3–5 engineering days |
| 1. Frozen spec registration | Register and inspect a PyTorch operator | Sandbox for reference/cases, artifact store | 1–2 weeks |
| 2. Immutable submissions | Submit local/Git implementation and see status | Snapshotting, RBAC, database | 1–2 weeks |
| 3. End-to-end A5 validation | Real kernel compiled and checked against PyTorch | A5 worker, CANN, runtime | 2–3 weeks |
| 4. Benchmark + global ranking | Per-case µs, geomean, publish validated results | Stable benchmarks and baseline | 1–2 weeks |
| 5. A3, source/binary delivery | Independent A3 rankings and compatible artifact retrieval | A3 worker, `cannery-kernels` loader | 2–3 weeks |
| 6. Harden / demonstrate | Repeatable PoC and operational controls | Security, backups, failure testing | 1–2 weeks |

Elapsed calendar time depends heavily on A3/A5 and build toolchain access. Parallelize `cannery-kernels` runtime development from milestone 1.

## Milestone 0 — Repository and service foundation

Deliverables:
- FastAPI `/health` and `/api/v1` routes, Jinja2/HTMX app shell, Typer client.
- PostgreSQL + Redis in Docker Compose, Alembic migrations and typed Pydantic DTOs.
- Local filesystem fake artifact store plus Hugging Face private storage adapter contract.
- GitHub Actions lint, unit tests and packaging checks on CPU, no device hardware requirements.
- Docs and accepted mockups under `docs/`.

Acceptance:
- `docker compose up` starts an empty local web catalog.
- `cannery --help` and a health-check CLI command work.
- API/UI/DB integration checks run in CI without NPU.

## Milestone 1 — Freeze a new PyTorch specification

Deliverables:
- `cannery register` from local path and pinned Git URL + commit + path.
- Strict YAML schema for reference, case generator, input/output contract, default tolerance, environment lock.
- Sandbox correctness smoke execution for the **reference itself**.
- Immutable snapshot, SHA-256 digest, unique `namespace/kernel@spec-vN` and versioned metadata.
- UI: Register New Kernel and Specification Detail.

Acceptance:
- Register `team/fa_xxx@spec-v1`, inspect the frozen manifest and source provenance.
- Attempts to overwrite or mutate a registered spec are rejected.
- A changed PyTorch reference or case generator creates spec-v2 rather than editing v1.

## Milestone 2 — Submit implementations against frozen specs

Deliverables:
- `cannery submit` for local snapshot or Git URL/commit/path; entrypoint, language, device and requested tolerance.
- Immutable source digest, parent revision link and author ownership.
- My Submissions page with queued status, source provenance and audit events.
- Basic worker capability registration and scheduler queue.
- Auth boundaries between users and admins; admin worker/queue view separate.

Acceptance:
- Multiple CATLASS/Triton/Ascend C candidates can target spec-v1 without re-supplying reference/cases.
- Git branches resolve to full SHA and repeatably restore snapshotted sources.
- Normal user cannot access worker or storage admin APIs.
- User receives stage status and an ETA range, or 'insufficient history' when no reliable estimate exists.

## Milestone 3 — A5 correctness vertical slice

Deliverables:
- One A5 worker with registered CANN/torch_npu/DSL toolchain versions.
- First end-to-end runtime/build adapter and isolated execution environment.
- Deterministic PyTorch reference and optimized implementation comparison.
- Per-case error summaries, first tolerance profile, failed-case logs.
- Durable device leases, retries and timeout behavior.

Acceptance:
- A5 candidate runs against canonical PyTorch spec.
- A deliberate incorrect implementation fails with actionable error.
- The UI shows validating → failed/validated without claiming finalized performance during validation.
- The worker survives restart without losing completed results or double-publishing artifacts.

## Milestone 4 — Benchmarks and global leaderboard

Deliverables:
- Warmup, synchronized device timing, repetition, median and p95 **µs**.
- Pinned PyTorch baseline, case-level reports and geomean speedup.
- Versioned built-in tolerance profiles; filter eligibility based on proven results.
- Explicit owner publish action; global page only shows validated + published results.
- Per-device overall best-so-far **step plot**, points colored by winning language, no cross-device ranking.
- User-facing benchmark history and projected job completion time.

Acceptance:
- Multiple candidate revisions compete under the same spec, A5, test suite and tolerance.
- A valid but unpublished candidate stays private.
- A published but accuracy-ineligible candidate does not appear for a stricter selected tolerance.
- History is a single best-so-far step line with language-colored record events.
- Per-case µs and geomean are independently reproducible from stored inputs.

## Milestone 5 — Additional device and distribution

Deliverables:
- A3 device worker and separate A3 leaderboard/history.
- Productionized `cannery-kernels` A3/A5 loader / variant matcher.
- Source and binary publication, ABI/toolchain manifests, checksums.
- Environment-aware download; explicit `--device`, `--lang`, `--source`, `--no-detect`.
- Ascend C, Triton-Ascend, CATLASS adapters as availability permits.

Acceptance:
- Same spec has A3 and A5 candidates; results cannot be mixed or directly compared in the ranking.
- User can download source regardless of local binary compatibility.
- Compatible binary is selected and executed through a `huggingface/kernels`-compatible API, or a clear compatibility diagnostic is returned.

## Milestone 6 — PoC hardening and demonstration

Deliverables:
- Worker failure recovery, idempotent jobs, upload consistency, artifact retention.
- AuthN/AuthZ, Git fetch allowlist/SSRF protection, sandbox and resource quotas.
- Logging/metrics, admin dashboard, backups and restore exercise.
- Demo runbook and sample user-defined kernel/spec with competing revisions.

End-to-end demo:
1. Register `team/fa_xxx@spec-v1` from a Git commit, freezing YAML/reference/cases.
2. Submit an A5 Triton candidate from a newer Git commit without copying the spec.
3. Observe estimated test time; validate against the immutable PyTorch reference.
4. Show µs timings and geomean speedup, publish explicitly.
5. Filter accuracy thresholds and show an A5 best-so-far step history with language-colored improvement markers.
6. Submit an A3 revision, show entirely separate rankings.
7. Download sources and the correct compatible binary, execute locally with the runtime.

## Cross-repository critical path

`cannery-kernels` work runs in parallel:
- R0: audit Hugging Face loader/packaging contracts and add minimal Ascend device detection.
- R1: fetch/load/call one pinned Ascend C binary end to end.
- R2: add CATLASS/Triton source/JIT paths and revisions/side-by-side isolation.
- R3: resolve A3/A5, CANN, torch_npu, Python ABI; provide stable errors.
- R4: integrate worker and CLI download, compatibility tests, upstream contribution assessment.

Do not block the platform's registration or submission UI waiting for all runtime backends: use a mock worker or a single validated A5 execution path first.

## Staffing and infrastructure

Suggested core team: 2 platform engineers (API/data/UI and worker/evaluation), 1 compiler/runtime engineer for `cannery-kernels`, and part-time validation/device infrastructure support. Minimal equipment: 8-vCPU/32-GB control-plane VM, one A3 and one A5 worker host, private Hugging Face storage, PostgreSQL and Redis. A CPU builder host is optional until compilation demand requires it.

## Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Arbitrary PyTorch code execution | High | Separate sandbox, no worker host secrets, restricted network/filesystem |
| Unstable device timing | High | Exclusive device lease, native timing, repetitions, environment cohort |
| Arbitrary tolerance filters | High | Versioned tested profiles; revalidate unseen pairs |
| Immutable Git refs vs mutable repos | Medium | Full SHA, snapshot and digest |
| Unsupported Ascend HF loader | High | Early hardware spike in `cannery-kernels` |
| Private artifact policy | Medium | Corporate approval; S3 adapter fallback |
| Variable queue ETAs | Medium | Rolling empirical stage estimates; show ranges / uncertainty |
| Fixed-suite model leaking into design | Medium | First-class user-provided reference and cases |

See [Design](design.md) and [Implementation](implementation.md).
