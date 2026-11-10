# Registering an immutable specification (M1)

M1 provides Git-only registration through the API, CLI and Register Kernel page.
M2 adds [source submissions and revisions](submissions.md). NPU builds, numerical
validation, benchmarking and publication are later milestones. Authentication currently uses an explicitly
enabled development identity; this is a local development deployment.

## Run locally

Build/start with `docker compose up --build -d --wait`. Enable the local identity
with `CANNERY_DEV_IDENTITY_ENABLED=true`; its namespace defaults to `team` and is
set with `CANNERY_DEV_NAMESPACE`. `CANNERY_DEV_ROLE=admin` permits semantic version
updates. Request headers cannot change namespace or role.

The trusted registrar needs the Docker socket and its dedicated snapshot volume.
The API and ordinary workers do not get the socket. The reference image is built
from `Dockerfile.smoke` and matched to the lock below. Author code receives only
retained source bytes in a resource-limited, network-disabled container, without
the Git token, Docker socket or database credentials.

Set `CANNERY_GIT_REMOTES` to a JSON list of exact HTTPS repository rules:

```json
[{"url":"https://github.com/huawei-cpl-zurich/helloworld-kernel.git","namespaces":["team"]}]
```

A rule can add `username` (default `oauth2`), `token`, and a registrar-local
`ca_file`. Private network endpoints require explicit `allowed_networks` CIDRs.
DNS resolution is bounded and approved addresses are pinned for the fetch.
Redirects, inherited proxies, SSH transport, submodules, symlinks and LFS pointers
are unsupported. Credentials belong in operator configuration, never manifests or
repository URLs. The test-only `compose.registration.yaml` uses synthetic private
Git credentials; do not use that fixture as deployment configuration.

## Source contract

The selected directory contains `cannery.yaml`, the declared Python files and a
JSON `environment.lock`. Dependencies outside that directory must be declared by
repository-relative paths using repeatable `--dependency` options. At most 256
regular files / 16 MiB are retained. A single trailing source-directory slash is
accepted; all paths must stay within the repository.

```yaml
schema_version: 1
kernel: {namespace: team, name: helloworld}
spec:
  version: 1
  reference: {file: reference.py, function: reference}
  cases: {file: cases.py, function: generate_cases}
  interface: {inputs: [x], outputs: [result]}
  validation:
    metric: torch_close
    equal_nan: false
    default_rtol: 0.00001
    default_atol: 0.000001
```

```json
{"schema_version":1,"python":"3.12","torch":"2.6.0+cpu"}
```

`reference.py` can define `def reference(x): return x + 1`. `cases.py` must provide a
callable yielding dictionaries of named inputs, for example:

```python
import torch

def generate_cases():
    yield {"x": torch.ones(2)}
```

The diagnostic requests up to three cases and tensor outputs matching the
interface. Installed Python/PyTorch must match the lock; author dependencies are
never installed. Arbitrary Python is supported, so these responses are
**diagnostics, not independent evidence of correctness or callable execution**.
See [reference diagnostic limits](reference-smoke.md).

## Register and observe

The [public hello-world repository](https://github.com/huawei-cpl-zurich/helloworld-kernel)
contains a verified spec at commit `99a8a1a9fd0d1b471e6520bafbbb1020fcdc198d`, directory `spec`.
It registers `team/helloworld@spec-v1` without shared dependencies. In the `team`
namespace, **Use hello-world example** on `/register-kernel` fills those pinned
values. The normal form stays blank until you choose the example; using it still
requires the exact public repository URL in the operator allowlist above.

For the local stack on port8095, start from your Cannery checkout:

```bash
CANNERY_PORT=8095 CANNERY_DEV_IDENTITY_ENABLED=true \
CANNERY_GIT_REMOTES='[{"url":"https://github.com/huawei-cpl-zurich/helloworld-kernel.git","namespaces":["team"]}]' \
docker compose up --build -d --wait
```

Preserve any other repository rules and operator settings. Then run on your host:


```bash
cannery register --repo https://github.com/huawei-cpl-zurich/helloworld-kernel.git \
  --commit 99a8a1a9fd0d1b471e6520bafbbb1020fcdc198d --path spec --url http://localhost:8095 --wait
cannery registrations --url http://localhost:8095
```

`POST /api/v1/kernels` accepts `{"source":{"type":"git","url":"...",
"commit":"99a8a1a9fd0d1b471e6520bafbbb1020fcdc198d","path":"spec","dependencies":[]}}` and
returns HTTP202 with an ID/status URL. `GET /api/v1/registrations/{id}` reports
queued/running/complete/failed; `GET /api/v1/kernels/team/helloworld/specs/1` reads
frozen metadata. The web form uses the same durable submission service. There is
no local-source archive upload endpoint.

PostgreSQL owns job state; Redis dispatch loss and expired worker leases are
reconciled with bounded retries. A failed job exposes a safe error code and can
be resubmitted after fixing its source or configuration. A pending CLI wait may
be observed again by ID through the status endpoint.

A successful registration retains selected files in protected internal Git
snapshot refs, with upstream SHA/path/tree mapping and content digests. It does
not mirror upstream history or unrelated files. PostgreSQL spec records reject
updates/deletes, including direct SQL. Repeating identical content is idempotent;
changed content requires the next sequential spec version and administrator
permission. Source is retained in Git, never uploaded to HF.

## CPU acceptance

Run `python scripts/registration_acceptance.py` with local Docker. It builds a
private HTTPS Git fixture and two registrars, executes actual CPU PyTorch
containers, checks CLI/API registration and immutable versions, injects service
outages and stale leases, deletes upstream and rebuilds from retained snapshots.
It owns and removes its project containers, images, networks and volumes. The
separate reference acceptance covers isolation, timeouts and output floods. No
A3/A5 hardware evidence is required or claimed for M1.
