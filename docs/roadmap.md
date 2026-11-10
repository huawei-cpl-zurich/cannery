# CANNery — PoC Roadmap

Current M3 contract: [private binary admission](binary-admission.md). The active
backend is self-hosted; no actual Hugging Face service is required. Ascend C A5
`unary-f32-v1` is the initial supported binary contract. CATLASS, A3, full
correctness/benchmarks and publication below remain proposed later milestones.


Status: M0–M2 source workflows implemented; M3 onward planned · 2026-11-10

Build a working user journey with thin end-to-end slices. **No non-Git source intake.** Every implementation commit contains a supported build contract, and binary artifacts are produced by our workers before validation and published from private artifact storage. **Initial adapters: Ascend C and CATLASS DSL only**; Triton-Ascend and PyPTO are future phases.

## Milestone overview

| Stage | Deliverable | Acceptance |
|---|---|---|
| M0 — foundations | FastAPI/Web/CLI, PostgreSQL, Redis, Git intake skeleton, fake artifact storage and CPU CI | Local services boot and health checks pass |
| M1 — frozen PyTorch spec | External Git fetch and **selected-subtree Git snapshots**; verified commit/path, schema, content hash; register spec-v1 | Register spec from Git; spec is immutable and rebuildable from internal snapshot commits |
| M2 — Git submissions | Save source binding, `submit --repo --commit --path`, `revise --commit`, user status/ETA; strict `submission.yaml` | Commit-only revision preserves prior submission and inherits frozen target |
| M3 — Ascend C binary admission | Adapter preflight/compile/package, A5 device build and loader, binary pushed to private content-addressed staging | Pinned Git commit produces actual loadable A5 binary with hash/build metadata |
| M4 — CATLASS binary admission | Ahead-of-publication CATLASS compilation and specialization/export proof; adapter and ABI metadata | Pinned CATLASS commit builds a loadable compatible artifact; unsupported specializations rejected |
| M5 — correctness & benchmark | Frozen PyTorch oracle; tolerance profile validation; synchronized µs timings; geomean | Bad binary fails; passing binary has complete per-case evidence |
| M6 — published leaderboard and consumption | Explicit publish, global vs personal submissions, device/tolerance filtering, private pinned loader integration | Published validated binary can be downloaded and loaded; single best-so-far step history with language-colored record points |
| M7 — A3 and hardening | A3 variants, separate rankings, sandbox/secret/backup and failure recovery | Repeat user journey on A3 and A5 without cross-device comparisons |

M3–M5 may be developed in parallel after M2. Testing the CATLASS exported-binary path should start as an **early technical spike** even before M3 completes. Work in `cannery-kernels` occurs alongside platform work.

## M0 — Foundation

- FastAPI API + server-rendered HTMX website, Typer CLI, SQLAlchemy/Alembic, PostgreSQL, Redis/Dramatiq, Docker Compose.
- CI: Ruff, Pytest, typing and packaging, with local bare Git fixture repositories.
- Private artifact service binary-artifact adapter and fake testing store.
- Reserve data volumes for PostgreSQL and **internal per-kernel Git snapshot repositories**.
- User navigation: Global Best, My Submissions, Register Kernel, Submit Implementation; worker/storage administration is role gated.

## M1 — Git-backed immutable specifications

- `cannery register --repo URL --commit FULL_SHA --path specs/fa_xxx/`.
- Restricted Git intake handles GitHub/GitLab/self-hosted, including private scoped credentials.
- Retain selected files in internal synthetic Git snapshots, protect refs against garbage collection and compute digests. Upstream commit/tree identities are provenance, without mirroring unrelated source/history.
- Read `cannery.yaml`, `reference.py`, `cases.py` and environment lock from the verified Git tree.
- Freeze `kernel@spec-v1` in PostgreSQL; any semantic spec update becomes spec-v2.
- Run a CPU reference diagnostic in an isolated worker; API process never executes arbitrary reference code. Arbitrary Python responses do not independently establish callable execution, numerical correctness or performance.
- See [implemented registration contract and CPU acceptance](registration.md).

**Acceptance:** source lives only in Git; no local source file upload API or HF source repository exists.

## M2 — Implementation submissions and commit-only revisions

- `submission.yaml` uses strict schema for device, language, entrypoint and `implementation.build` adapter/options.
- `cannery submit` accepts Git URL, pinned SHA and subdir; rejects non-Git upload or unsupported adapter IDs.
- `cannery revise SUB_ID --commit FULL_SHA` inherits repo, path, spec, device, language and build adapter while creating a new immutable record.
- Intake imports only selected source directory and declared dependencies as a new internal Git snapshot commit; retains the upstream SHA mapping without mirroring upstream history.
- My Submissions shows queue, build, correctness, benchmark and completion state, with empirical ETA or honest unknown.

**Acceptance:** two different Git SHAs create two immutable candidate revisions, without re-registering or copying the spec.

Implemented source intake ends at `awaiting_build`; build/correctness/benchmark are
pending and ETA is unknown. Author implementation code is never executed during M2.
See [submission usage and CPU failure acceptance](submissions.md).

## M3 — Ascend C binary build

- Select A5 worker with matching CANN compiler/runtime.
- Implement typed `AscendCAdapter.preflight/build/package`.
- Produce shared-library or runtime-compatible binary plus generated loader metadata, hashes and ABI/toolchain fingerprint.
- Upload the generated artifact to private content-addressed staging; PostgreSQL indexes pinned artifact hash.
- Verify that a minimal `cannery-kernels` runtime can load/call this binary on A5.
- Compilation logs and failures are private job records.

**Acceptance:** submit Git source → compiled, stored and executed A5 Ascend C binary. Precompiled author binary alone is not admissible.

## M4 — CATLASS DSL binary build

- Determine actual supported CATLASS compiler export/JIT specialization mechanisms.
- Define supported shapes/dtypes specialization manifest.
- Implement `CatlassAdapter` with controlled ahead-of-publication compilation and exported loadable artifacts.
- Validate compatible runtime wrapper and same loader path as Ascend C where feasible.
- Reject pure JIT-only unsupported variants; do not claim a packaged binary exists where it does not.

**Acceptance:** a pinned CATLASS implementation commit yields reproducible loadable binary candidate and manifest for its supported cases.

## M5 — Correctness and performance evaluation

- Match worker device, language, toolchain versions and target ABI.
- Run PyTorch reference and built binary against frozen mandatory cases; persist case-level errors and tolerance-profile outcomes.
- Pin reference baseline, capture warmup and synchronized NPU timings in **µs**, compute speedup per case and geometric mean across all cases.
- Version spec/suite/device/tolerance/environment cohorts. Evaluate untested tolerance profiles on demand.
- Save run logs/raw timing and generated binaries in private artifact; outcomes and searchable aggregates in PostgreSQL.

**Acceptance:** failed correctness never produces a globally eligible performance score; valid submissions are privately inspectable.

## M6 — Release, leaderboard and consumer

- Explicit publication by owner/authorized reviewer after successful build + correctness + full benchmarks.
- Package the validated binary into the private release layout; pin exact content hashes.
- CANNery CLI resolves compatible binaries by device/runtime; no user source copy in HF.
- The private runtime loader loads and calls a published A5 binary.
- Global Best filters only published, valid, tolerance-eligible results; My Submissions retains failures and private work.
- One **overall best-so-far** step line per device, markers colored by record-setting language.

**Acceptance:** end user finds, selects and loads binary from private artifact using pinned revision, while source inspection points to external Git SHA/path.

## M7 — A3 and operations

- Add A3 worker and matching Ascend C/CATLASS binary variants; completely separate A3 rankings/history.
- Apply device leases, idempotent dispatch, heartbeat recovery, restricted Git intake, sandbox execution, credential scoping and artifact cleanup.
- Back up **PostgreSQL and internal Git snapshot repositories**, test restore, reconcile orphaned private staging artifacts.
- Role gate workers/storage/Git snapshot admin UI; expose only user-relevant submission ETA.

## Parallel `cannery-kernels` deliverables

1. Confirm private package metadata and ABI contract for CANN backend.
2. Load one pinned Ascend C binary in A5; record exact loader and trust behavior.
3. Add CATLASS exported binary/adapter loading; handle supported specialization keys.
4. Provide A3/A5 and CANN/torch_npu/Python compatibility matching and pinned artifact hashes.
5. Contribute relevant Ascend support upstream when independently reusable.

## Beyond PoC

Triton-Ascend and PyPTO become additional **versioned build adapters** with independent early compiler-export feasibility checks. Git webhooks can automatically submit each new commit to a bound implementation track, but must follow the same admission, validation and publication gates.

## Infrastructure and risks

Starting resources: control-plane VM 8 vCPU/32 GB with PostgreSQL + Redis; persistent **Git snapshot volume initially budgeted at 20–50 GB**, adjusted to measured imported file sizes and separate DB/temporary storage; one A3 and one A5 worker host with sufficient CANN/toolchain scratch; private binary/results storage ~100–500 GB. Scale mirror, artifacts and worker disk based on measured source and compiled artifact sizes.

Main risks: CATLASS specialized JIT binaries may not be generally exportable, Ascend C callable ABI packaging, arbitrary Git source security, correct device timing, and private artifact permissions. Gate milestones on real compile→store→load→execute demonstrations rather than mocked binaries.

See [System Architecture](architecture.md), [Implementation](implementation.md) and [User Story](user-story.md).

### Revised Git storage acceptance

The importer accepts a potentially large upstream monorepo but persists only the selected kernel/spec directories and declared dependency files as compact internal Git snapshot commits. Submitting only a new SHA creates an internal child commit and supports diffs across accepted revisions. The system does not retain the upstream DAG, must keep the upstream/internal SHA mapping, and must prove an internal snapshot can build after deleting the temporary upstream checkout.
