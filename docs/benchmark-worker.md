# Private benchmark worker

Enable benchmarking after the artifacts, builds, and numerics overlays:

```bash
docker compose --env-file ~/.config/cannery/service.env -p cannery \
  -f compose.yaml -f compose.artifacts.yaml -f compose.builds.yaml \
  -f compose.numerics.yaml -f compose.benchmarks.yaml up -d --build --wait
```

The opt-in benchmarker discovers existing and future numerically passed candidates.
Failed correctness and legacy specs are excluded. Results remain private; this
milestone does not publish releases or create leaderboard entries.

`sync-call-v1` uses 20 warmups and 100 synchronized wall-clock call samples per
nonempty case, in microseconds. Input preparation, transfers and package loading
are excluded. Callable launch, output allocation and completion are included.
Empty cases remain mandatory correctness checks but do not affect the score.
An entirely empty suite completes without a speedup score.

A shared A5 PyTorch timing baseline is generated once per frozen spec/suite,
recipe generator, timing protocol, physical device, runtime image/fingerprint,
driver and loader cohort. The existing CPU output cache remains the correctness
oracle and is never regenerated for performance measurement. Both reference and
candidate outputs are verified against it before and after timing. Candidate
jobs never receive golden outputs.

All nonempty mandatory cases must have finite positive samples. Reports show
median and nearest-rank p95 latency, reference medians, per-case speedups and the
geometric mean speedup. Raw timing samples use private, digest-pinned artifacts.
A failed or partial run never produces an accepted score.

The worker inherits the registered profile, transport, pinned loader, process
reaping and private storage settings from the builder. Trusted supervision checks
physical identity and healthy idle isolation before and after timing under the
shared device lease. Busy devices defer; changed cohorts block rather than mix
measurements. Reference-cache outages never rerun a completed baseline.

Retained receipts/checkpoints resume the same attempt after restart or transport
loss. Jobs have a 30-minute deadline and at most three attempts; remote execution
is bounded to 180 seconds. Preserve control/artifact volumes across upgrades and
drain active remote work before changing the supervision contract.
