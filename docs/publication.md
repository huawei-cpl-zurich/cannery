# Publication and offline consumption

M6 adds explicit publication for completed A5 submissions. Results stay private
until their owner (or the development administrator) publishes them. Valid
kernels slower than PyTorch are eligible. Publication reuses retained evidence;
it does not build, run a baseline, validate a new tolerance or benchmark again.

## Enable the services

Use the normal build/numerical/benchmark configuration and append the release
overlay. Keep the operator environment file private:

```bash
docker compose --env-file ~/.config/cannery/service.env \
  -f compose.yaml -f compose.artifacts.yaml -f compose.builds.yaml \
  -f compose.numerics.yaml -f compose.benchmarks.yaml -f compose.releases.yaml \
  up -d --build
```

The overlay enables publication API/web routes and a publisher with PostgreSQL
and private artifact access. The publisher has no Docker socket, SSH credentials
or NPU access. The API only receives the artifact read token. Apply migrations
through the Compose `migrate` service before starting dependent services. Keep
existing PostgreSQL, snapshot and artifact volumes. Do not use the registration
acceptance overlay for the normal service.

This PoC uses the configured development namespace/role, not production
multi-user authentication. Keep the service within its intended development
access boundary. No real Hugging Face service or repository is needed.

## Publish, observe and withdraw

The submission detail page offers **Publish release** after a passing benchmark.
It shows private/publishing/published/failed/withdrawn visibility and polls while
publication is pending. The server independently checks all eligibility evidence.

```bash
cannery publish SUBMISSION_UUID --url http://127.0.0.1:8095 --wait
cannery publication-status SUBMISSION_UUID --url http://127.0.0.1:8095
cannery withdraw SUBMISSION_UUID --url http://127.0.0.1:8095
```

The REST equivalents are `POST /api/v1/submissions/{id}/publish` and `/withdraw`,
with JSON `{}`. Status is `GET /api/v1/submissions/{id}/publication`; an accepted
publish returns its durable job ID at `/api/v1/publication-jobs/{id}`.
A repeated active request returns the same job. If a reply is lost, inspect the
submission status before retrying. State-changing CLI calls do not automatically
retry an ambiguous response; read-only resolution/downloads use bounded retries. Jobs have a ten-minute deadline, three bounded
attempts and exponential backoff. A failed publication can be requested again.

The worker verifies binary metadata/digests, saved timing samples and immutable
manifest storage before SQL makes the release visible. Generation and lease
fences prevent an old worker from publishing after withdrawal. Withdrawal
cancels pending work and removes discovery, rankings, chart points and new
manifest/package downloads. It retains the audit and already downloaded bytes.
Republishing restores the same immutable release ID and manifest/package hashes
with a new activation event and publication time.

## Global Best and saved evidence

Open `/global-best` and select a kernel. The default is its latest frozen spec;
it never silently falls back to an older published spec. Select an explicit spec
version to inspect older releases. Multiple measurement cohorts require an
explicit cohort hash. Cohorts separate the suite, device, runtime, environment,
driver, image, loader and timing protocol. Only an exact saved default/requested
`rtol,atol` pair is supported; supply both or leave both blank for spec defaults.
There is no tolerance inference or on-demand correctness run.

Rankings use geometric mean speedup over nonempty cases and include slower valid
kernels. Empty cases remain correctness-only. All-empty releases have no score:
they appear in release metadata and allow explicit pinned consumption, but never
win automatic resolution. The chart is one strict best-so-far curve in publication
time with language markers. Currently withdrawn releases are excluded. This is
synchronized call latency (`sync-call-v1`), not device-only kernel latency.

Release pages show pinned Git source, exact runtime/ABI/static shapes, validated
tolerances, package/manifest hashes, per-case median/p95 latency and original
saved numerical comparison errors. The original and benchmark before/after
correctness checks must all pass eligibility. Private worker logs, backend
credentials and internal artifact URLs are not exposed in metadata.

```bash
cannery info team/helloworld --url http://127.0.0.1:8095
cannery info team/helloworld --spec-version 2 --cohort COHORT_SHA256 \
  --limit 50 --offset 0 --url http://127.0.0.1:8095
cannery source RELEASE_UUID --url http://127.0.0.1:8095
```

Metadata endpoints: `/api/v1/releases/kernels`, `/api/v1/releases?kernel=...`,
`/api/v1/releases/{id}`, `/api/v1/leaderboard?kernel=...` and
`/api/v1/leaderboard/history?kernel=...`. Listings support bounded `limit`/`offset`.
The release listing includes scoreless releases; the leaderboard excludes them.

## Resolve and download

Inside the consumer's configured A5 runtime with the offline SDK installed
(runtime detection runs in a bounded 30-second subprocess):

```bash
cannery get team/helloworld --shape 2,3 --language ascendc \
  --url http://127.0.0.1:8095
cannery get team/helloworld --release-id RELEASE_UUID \
  --url http://127.0.0.1:8095
```

Automatic resolution chooses the highest-scoring currently published compatible
release. Compatibility requires the exact six-field runtime fingerprint,
supported language, device, spec/cohort/tolerance and supported shape. Ascend C
supports dynamic float32 shapes; CATLASS uses the declared static shapes. An
explicit release pin can select an older spec and never silently changes release
when a new spec freezes. Incompatible pins fail rather than falling back.

On a CPU host, disable detection and supply a small JSON file containing the
consumer's exact `machine`, `python`, `torch`, `torch_npu`, `cann` and `soc` values:

```bash
cannery get team/helloworld --no-detect --runtime-json runtime.json \
  --release-id RELEASE_UUID --cache ./download-cache \
  --url http://127.0.0.1:8095
```

`POST /api/v1/artifacts/resolve` accepts the same selection and runtime as JSON.
Authenticated installation consumers download bytes through
`/api/v1/releases/{id}/manifest` and `/package`; backend credentials remain
internal. The API rechecks publication state after backend reads. The CLI
verifies manifest/package hashes and metadata, applies bounded transfer/retry
budgets and atomically writes a content-addressed cache. Existing cache bytes
are verified before reuse. Redirects and encoded HTTP responses are rejected.
The command prints the local paths and immutable hashes and executes no code.

Use the downloaded bytes in a matching isolated runtime with the existing SDK:

```python
from pathlib import Path
from cannery_kernels import load_cannery_package

package = Path("PACKAGE_PATH_FROM_GET").read_bytes()
kernel = load_cannery_package(package, "PACKAGE_SHA256_FROM_GET", trust_remote_code=True)
y = kernel.forward(x)
```

Native packages execute code: the trust flag is explicit. Loading is fully
offline, performs its own digest/runtime checks and remains possible with saved
bytes after withdrawal. A new CLI resolution/download still checks current
publication visibility. No online Python client or dependency fetching is added.

## Validation

Relevant local checks include `pytest tests/test_release*.py tests/test_publication*.py`.
Real local fault acceptance:

```bash
CANNERY_RELEASE_DOCKER=1 pytest -q tests/test_publication_docker.py
```

That CPU-only test uses two publisher containers, PostgreSQL and private HTTP
storage. It injects process loss after upload, unavailable storage, delayed
publication and withdrawal races; it verifies deduplication, bounded recovery
and immutable republish identity and cleans its own containers, volumes and
image. It does not establish device correctness. Fresh consumer acceptance on
A5 must load/call both Ascend C and CATLASS packages through the offline SDK and
compare seeded outputs with the cached CPU baseline. Publication itself requires
no hardware run and never regenerates that baseline.
