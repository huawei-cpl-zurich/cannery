# CANNery

CANNery is an Ascend-oriented kernel registry, benchmarking, and distribution platform for user-defined PyTorch operators.

Authors register a frozen, versioned, Git-backed PyTorch specification and subsequently submit Git commit revisions with mandatory build manifests in Ascend C or CATLASS DSL initially, with Triton-Ascend and PyPTO planned later. CANNery compiles binary candidates, retains selected source files in internal Git snapshots, stores generated binaries in private Hugging Face repositories, and validates correctness, measures per-case latency in microseconds, publishes explicit opt-in results, and ranks them independently by device and accuracy requirements.

## Documentation

- [System design](docs/design.md) — scope, immutable specs, submissions, tolerances, leaderboards, UI mockups and infrastructure.
- [System architecture](docs/architecture.md) — deployable services, sequence diagrams, data landing/copying and Hugging Face artifact lifecycle.
- [User story](docs/user-story.md) — website, CLI, REST submissions and Hugging Face loading/execution.
- [Implementation notes](docs/implementation.md) — Python libraries, API/data contracts, scheduling, benchmark engine and security.
- [Roadmap](docs/roadmap.md) — incremental PoC milestones, acceptance criteria and cross-repository runtime work.

## Related repository

- [cannery-kernels](https://github.com/huawei-cpl-zurich/cannery-kernels) — Ascend extensions to Hugging Face's kernel loader/runtime.

## Status

M0 foundations and M1 Git-backed specification registration are implemented.
The API, CLI and web form register immutable versions through durable workers,
retain selected Git snapshots and run isolated CPU reference diagnostics. See
[registration](docs/registration.md) for source format, configuration and usage,
and [local development](docs/development.md) for setup and validation.
Implementation submission, NPU builds, numerical validation and publication are
later milestones. CPU diagnostics do not independently prove correctness.
