# CANNery

CANNery is an Ascend-oriented kernel registry, benchmarking, and distribution platform for user-defined PyTorch operators.

Authors register a frozen, versioned PyTorch specification and subsequently submit any number of implementation revisions in Ascend C, Triton-Ascend, or CATLASS DSL. CANNery validates correctness, measures per-case latency in microseconds, publishes explicit opt-in results, and ranks them independently by device and accuracy requirements.

## Documentation

- [System design](docs/design.md) — scope, immutable specs, submissions, tolerances, leaderboards, UI mockups and infrastructure.
- [Implementation notes](docs/implementation.md) — Python libraries, API/data contracts, scheduling, benchmark engine and security.
- [Roadmap](docs/roadmap.md) — incremental PoC milestones, acceptance criteria and cross-repository runtime work.

## Related repository

- [cannery-kernels](https://github.com/huawei-cpl-zurich/cannery-kernels) — Ascend extensions to Hugging Face's kernel loader/runtime.

## Status

Design stage. The Markdown docs and SVG image previews are committed; UI and backend code remain to be implemented.
