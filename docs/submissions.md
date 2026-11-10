# Git submissions and revisions (M2)

M2 retains implementation source against an existing frozen specification. API,
CLI and web support initial submissions, commit-only revisions and private status.
Successful source intake ends at **awaiting_build**. Build, correctness and benchmark
stages remain pending; ETA is **unknown**. Compiler preflight, binaries, NPU runs and
publication belong to M3 and later milestones.

## Prerequisites and local startup

Register a specification first using [the M1 guide](registration.md). Your configured
development namespace must own that spec, and the operator must allowlist the exact
HTTPS implementation repository URL for that namespace through `CANNERY_GIT_REMOTES`.
Public repositories also require an allowlist rule. No Git archives or HF source
uploads are accepted.

After updating Cannery, rebuild the local stack with your existing operator values:

```bash
CANNERY_PORT=8095 CANNERY_DEV_IDENTITY_ENABLED=true \
CANNERY_GIT_REMOTES='[{"url":"https://github.com/huawei-cpl-zurich/helloworld-kernel.git","namespaces":["team"]}]' \
docker compose up --build -d --wait
```

Run that command from your Cannery checkout. The example repository supplies both the M1
spec and a host-tested M2 toy submission. Use your own implementation directory
for subsequent compiler and device validation. Preserve your existing rule list, identity, port and other operator
settings when rebuilding. Local development identity is configured server-side;
request headers cannot grant a namespace or administrator role.

## Implementation source contract

The selected directory contains `submission.yaml` and every declared build input.
The public example includes a toy under `candidates/helloworld`, with this manifest:

```yaml
schema_version: 1
kernel: team/helloworld
spec_version: 1
implementation:
  device: a5
  language: ascendc
  entrypoint: kernel.cpp:forward
  build:
    adapter: ascendc
    inputs: [kernel.cpp]
    artifact: helloworld_a5
    options:
      optimization: release
validation:
  requested_rtol: 0.00001
  requested_atol: 0.000001
```

At commit `99a8a1a9fd0d1b471e6520bafbbb1020fcdc198d`, `kernel.cpp` is a portable C++
add-one loop with host arithmetic tests. It is an M2 source-intake fixture using the
recognized `ascendc`/`a5` declaration, not an Ascend device kernel or NPU-validated
binary. After registering `team/helloworld@spec-v1`, **Use hello-world toy submission**
on `/submit-implementation` selects that frozen spec and fills the pinned source
and target. The preset is unavailable until the required spec is retained.

For your own implementation, use the namespace/name/version of your actual spec. M2 checks its declaration and retained bytes without
importing, executing or compiling it. Requested tolerances are recorded claims
until evaluated in a later milestone.

Recognized target declarations are devices `a3`/`a5` and languages/adapters
`ascendc`/`catlass`; adapter and language must match. Optimization is `release`
(default) or `debug`. Arbitrary build commands/options and unknown fields are
rejected. CATLASS declared inputs use `.py`; Ascend C inputs use `.c`, `.cc`, `.cpp`,
`.cce`, `.h` or `.hpp`. The entrypoint uses a relative file and symbol separated by
one colon, and its file must appear in `build.inputs`. Declared prebuilt input
extensions such as `.so`/`.o` are rejected. Compiler/export compatibility is checked
by later build adapters.

Build input paths are relative to the selected source directory. Files outside it
must be explicitly retained using repository-relative `--dependency` paths. The
Git intake limits remain 256 regular files / 16 MiB, with confined paths, no
symlinks, submodules, LFS pointers, hooks or inherited credentials/proxies.

## Submit and observe

From an installed Cannery CLI:

```bash
cannery submit team/helloworld --spec-version 1 --device a5 --lang ascendc \
  --repo https://github.com/huawei-cpl-zurich/helloworld-kernel.git \
  --commit 99a8a1a9fd0d1b471e6520bafbbb1020fcdc198d --path candidates/helloworld/ \
  --url http://localhost:8095 --wait
cannery submissions --url http://localhost:8095
cannery submission SUBMISSION_UUID --url http://localhost:8095
```

Inside Compose use `docker compose exec -T api cannery ... --url http://api:8000`.
`--wait` waits for source intake and reports awaiting build; a timeout leaves the
job running and prints its ID for subsequent inspection.

In the web UI, open `/submit-implementation`, choose an owned frozen spec and
enter target, Git URL, full SHA, source directory and dependency paths. The status
page polls queued/intaking work. `/my-submissions` lists your requests, private
failures, stage and unknown ETA. A retained candidate remains pending for build;
it has no numerical result, performance score or publication eligibility.

The initial REST request is `POST /api/v1/kernels/team/helloworld/submissions`:

```json
{"spec_version":1,"device":"a5","language":"ascendc","source":{"type":"git","url":"https://github.com/huawei-cpl-zurich/helloworld-kernel.git","commit":"99a8a1a9fd0d1b471e6520bafbbb1020fcdc198d","path":"candidates/helloworld","dependencies":[]}}
```

It returns HTTP202 with `id` and `status_url`. Read
`GET /api/v1/submissions/{id}` or `GET /api/v1/submissions/mine`. Namespace access
is enforced on every request. Invalid or unsupported request fields are rejected;
database outages return a safe 503. Redis dispatch loss leaves a durable queued row
that reconciliation republishes.

## Commit-only revision

```bash
cannery revise SUBMISSION_UUID --commit NEW_FULL_SHA \
  --url http://localhost:8095 --wait
```

REST: `POST /api/v1/submissions/{id}/revisions` with `{"commit":"NEW_FULL_SHA"}`.
The web detail page exposes the same full-SHA revision form after source retention.
A parent must have a retained candidate. Failed initial intake can be retried as a
fresh initial submission after fixing its cause.

Each revision gets a new immutable request/candidate and records its parent and
root binding ID. It inherits repository URL, source directory, dependency paths,
frozen spec, device, language, entrypoint and adapter. The new commit's manifest
must agree with those fields. Its source files, build inputs/options and requested
tolerances are retained as that revision's own immutable declaration. To change a
frozen binding field, make a fresh initial submission. Prior revisions never change.

Intake rereads and authenticates protected Git snapshots before freezing a candidate.
PostgreSQL guards reject updates/deletes even through direct SQL. Leases are checked
before and after snapshot verification; stale attempts cannot publish. Retries,
overall deadlines and scratch ownership are durable and bounded. Failed cleanup is
reconciled separately so it cannot starve ready work.

## Local failure acceptance

`python scripts/submission_acceptance.py` owns an isolated Docker project with a
private HTTPS Git fixture, PostgreSQL/Redis and two registrars. It exercises actual
CLI/API/web submission and revision, immutable source bindings, duplicate deliveries,
invalid source/manifest/target requests, Git/Redis/DB outages, abandoned leases and
shared scratch recovery. It deletes upstream, runs Git GC and revalidates retained
candidate sources, then cleans project containers, images, networks and volumes.
Candidate fixture code raises if executed. These are source-intake proofs; no NPU
build or numerical/performance validation is claimed. A3/A5 hardware is unnecessary.

### Initial A5 native build contract (M3)

Native admission requires `implementation.build.abi: unary-f32-v1`, device `a5`
and language/adapter `ascendc`. Omitting the ABI retains the source-only M2
behavior; a host C++ fixture is not silently interpreted as an NPU kernel.
The ABI is part of the frozen revision binding and cannot change through revise.
Existing manifest hashes are preserved when this optional field is absent/null.

The entry point is one declared Ascend C translation unit plus optional declared
headers. Export the selected device symbol with arguments
`(GM_ADDR input, GM_ADDR output, uint64_t count)` for contiguous float32 buffers.
The trusted adapter generates a one-block launch wrapper and fixed A5 compiler
arguments. It never interprets build inputs as commands or runs author build
scripts. Compilation must occur in the dedicated device-less sandbox; returning
a build plan alone does not prove compilation or device execution.
