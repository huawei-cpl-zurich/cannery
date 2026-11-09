# CANNery

CANNery is an Ascend-oriented kernel registry, benchmarking, and distribution platform for user-defined PyTorch operators.

Authors register a frozen, versioned, Git-backed PyTorch specification and subsequently submit Git commit revisions with mandatory build manifests in Ascend C or CATLASS DSL initially, with Triton-Ascend and PyPTO planned later. CANNery compiles binary candidates, retains accepted Git commits in an internal bare mirror, stores generated binaries in private Hugging Face repositories, and validates correctness, measures per-case latency in microseconds, publishes explicit opt-in results, and ranks them independently by device and accuracy requirements.

## Documentation

- [System design](docs/design.md) — scope, immutable specs, submissions, tolerances, leaderboards, UI mockups and infrastructure.
- [System architecture](docs/architecture.md) — deployable services, sequence diagrams, data landing/copying and Hugging Face artifact lifecycle.
- [User story](docs/user-story.md) — website, CLI, REST submissions and Hugging Face loading/execution.
- [Implementation notes](docs/implementation.md) — Python libraries, API/data contracts, scheduling, benchmark engine and security.
- [Roadmap](docs/roadmap.md) — incremental PoC milestones, acceptance criteria and cross-repository runtime work.

## Related repository

- [cannery-kernels](https://github.com/huawei-cpl-zurich/cannery-kernels) — Ascend extensions to Hugging Face's kernel loader/runtime.

## Status

M0 CPU foundation implemented: API/web/CLI, PostgreSQL/Redis worker plumbing, artifact
stores, Git intake contracts and fault-tested Docker acceptance. See
[Local development](docs/development.md) for setup and validation. Kernel intake,
NPU build/validation and publication remain later milestones.
