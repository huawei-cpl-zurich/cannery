# Numerical validation worker

Add `-f compose.numerics.yaml` after the artifacts and builds overlays to enable
recipe-based CPU ingress caching and A5 numerical validation. The registrar
requires a private artifact reader and writer, the existing pinned CPU reference
image, Docker socket and its dedicated `baseline-control` named volume. Both the
registrar and oracle jobs use that same volume through selected subpaths; do not
substitute a registrar-only bind mount.

From the Cannery checkout, with the existing operator environment configured:

```bash
docker compose --env-file ~/.config/cannery/service.env -p cannery \
  -f compose.yaml -f compose.artifacts.yaml -f compose.builds.yaml \
  -f compose.numerics.yaml up -d --build --wait
```

Inspect numerical results with `cannery submission SUBMISSION_UUID --url
http://localhost:8095` (use your configured API URL). `--wait-build` observes
binary admission; the status may already have advanced to numerical validation.

The validator inherits the trusted builder image, registered worker profile,
remote-access skill, SDK wheel and private storage settings. Its durable journals
live in `build-control/numerics`. It never gives CPU baseline outputs to the
candidate process. Device selection, leases, health checks and failure handling
remain guarded by the registered A5 job supervisor.

Each recipe spec gets a shared CPU baseline before it freezes. The cache binds
the spec digest, recipe suite, generator and locked CPU environment. Only output
tensors and provenance are stored. Re-registering an identical spec reuses that
artifact; changing the configured image does not regenerate completed baselines.
A missing or corrupt ready artifact blocks validation and needs operator repair.
No automatic baseline regeneration is performed.

Legacy Python-generator specs remain available, with numerical validation blocked
as `recipe_baseline_required`. Register a new recipe version to enable numerical
checks. Both the spec default and submission-requested `atol + rtol * abs(ref)`
checks must pass, along with unchanged input hashes. Numerical success means
`awaiting_benchmark`. The optional [benchmark worker](benchmark-worker.md) advances
passing submissions to private latency/speedup reports and `complete`. Publication
remains a later milestone.

Before upgrading the worker supervision contract, drain active binary builds and numerical validations.
Apply database migrations, then restart services while preserving all data and
control volumes. A coordinator restart resumes the retained attempt and remote
handle; uncertain operations are observed rather than resubmitted.
