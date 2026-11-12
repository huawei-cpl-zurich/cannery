# CANNery

CANNery is an Ascend-oriented kernel registry, benchmarking, and distribution platform for user-defined PyTorch operators.

Authors register a frozen, versioned, Git-backed PyTorch specification and subsequently submit Git commit revisions with mandatory build manifests in Ascend C or CATLASS DSL initially, with Triton-Ascend and PyPTO planned later. CANNery compiles binary candidates, retains selected source files in internal Git snapshots, stores generated binaries in a self-hosted private artifact backend, and load/call smokes them on A5. Correctness scoring, per-case latency, publication and rankings are planned next.

## Documentation

- [System design](docs/design.md) — scope, immutable specs, submissions, tolerances, leaderboards, UI mockups and infrastructure.
- [System architecture](docs/architecture.md) — deployable services, sequence diagrams, data landing/copying and artifact lifecycle.
- [User story](docs/user-story.md) — website, CLI, REST submissions and runtime consumption.
- [Implementation notes](docs/implementation.md) — Python libraries, API/data contracts, scheduling, benchmark engine and security.
- [Roadmap](docs/roadmap.md) — incremental PoC milestones, acceptance criteria and cross-repository runtime work.

## Related repository

- [cannery-kernels](https://github.com/huawei-cpl-zurich/cannery-kernels) — Ascend extensions to Hugging Face's kernel loader/runtime.

## Status

M0–M5 are implemented for supported A5 recipes: foundations, Git registration and
revisions, Ascend C/static CATLASS binary admission, seeded numerical validation
and private performance reports.
The API, CLI and web form register immutable versions through durable workers,
retain selected Git snapshots and run isolated CPU reference diagnostics. See
[registration](docs/registration.md) for source format, configuration and usage,
[submissions and revisions](docs/submissions.md) for M2 usage, and
[local development](docs/development.md) for setup and validation.
Opt-in native submissions compile in isolated workers, round-trip through private
storage and pass an A5 load/call smoke before `awaiting_validation`. See
[Ascend C setup](docs/binary-admission.md) and [CATLASS setup](docs/catlass-admission.md).
Source-only submissions remain `awaiting_build` with unknown ETA. Recipe-based
numerical validation caches CPU reference outputs once and regenerates seeded
inputs for each candidate. Opt-in [private benchmarking](docs/benchmark-worker.md)
adds synchronized call latency and speedup reports against a cached A5 PyTorch
baseline; passing candidates finish at `complete`. Legacy specs require recipe
migration. Publication and leaderboards remain M6. CPU diagnostics and binary
loadability alone do not prove correctness.
