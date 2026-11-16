# Private Hub consumption with get_kernel

Cannery exposes a small authenticated Hugging Face kernel-repository read API.
Our `cannery-kernels` fork uses the standard Hugging Face HTTP client to retrieve
published binary packages from this service. Nothing is hosted on huggingface.co.
Source remains in the original Git repositories; this interface downloads verified
binaries, not author Python or build scripts.

## Consumer setup

In the matching Ascend Python/PyTorch/torch-npu/CANN environment, install the fork
and its private runtime. From a `cannery-kernels` checkout:

```sh
pip install ./kernels ./cannery-runtime 'huggingface-hub==1.33.0'
cannery login --url http://localhost:8095 --username alice
export HF_ENDPOINT=http://localhost:8095
```

Use the same service origin for login and `HF_ENDPOINT`, including host and port.
The fork reads the private origin-bound credentials saved by `cannery login`.
Explicit `HF_TOKEN` overrides that file; an unrelated cached Hugging Face token
is never used for a private endpoint. HTTPS is required except on loopback.
Login still requires only username/password; an assigned namespace is automatic.

The release detail API (`GET /api/v1/releases/<uuid>`) includes a `hub` object
with repository, canonical revision and immutable `release-<uuid>` alias.
An operator must backfill older releases before this field is populated.

```python
import torch
from kernels import get_kernel

kernel = get_kernel(
    "team/helloworld",
    revision="release-<published-release-uuid>",
    backend="cann",
    trust_remote_code=True,
)
x = torch.randn(2, 3, device="npu", dtype=torch.float32)
y = kernel.forward(x)
torch.npu.synchronize()
```

Select a release matching the entire runtime fingerprint and device. A3 native,
A5 native and A5 CATLASS packages remain distinct; CATLASS inputs must match its
published specializations. This path never compiles or installs dependencies.
It executes native code, so use the same restricted runtime and device-isolation
policy as other Cannery consumers. Keep the module alive until its calls finish.

## Operator activation

Back up and drain the deployment using the [maintenance procedure](maintenance.md).
Apply migration0016 before starting updated API/publication workers, then run:

```sh
cannery hub backfill
```

The command is idempotent and creates mappings only; it does not regenerate
baselines, publish releases, copy artifacts or schedule kernel work. New
publications create mappings atomically. Enable `CANNERY_HUB_ENABLED=true` in the
API deployment after backfill; it is off by default and also requires releases
to be enabled. Existing accounts and service credentials are reused.

Supported read endpoints are kernel repository information at
`/api/kernels/<namespace>/<name>/revision/<revision>` and file HEAD/GET at
`/kernels/<namespace>/<name>/resolve/<revision>/<filename>`. The virtual snapshot
contains only `release.json` and `kernel.tar`; downloads support byte ranges.
Forty-character revisions identify the immutable projection; SHA256 pins verify
its manifest/package bytes. Repository and release identity must agree.

Every load rechecks publication online, including a warm cache. Withdrawal,
revoked credentials and backend failure fail closed. Downloads run in a separate
process with a 60-second deadline, including cache-lock and retry waits; that
process cannot change the caller's Hugging Face HTTP client. Native loading
remains in the caller and preserves the existing runtime/specialization checks.

This is not a full Hub implementation. Uploads, Git/LFS operations, moving `main`,
version/Global Best selection and offline Cannery get_kernel are unsupported.
Ordinary upstream kernel loading and explicit local overrides retain their
existing paths. Private package downloads ignore ambient proxy settings and
reject credential forwarding to other origins.

## Website quickstart

The footer **Help** page explains client installation, sign-in and running a
loaded kernel. Each published release has a split **Download** button: the main
action saves the verified binary archive; the arrow offers pinned Python/Hugging
Face loading and CLI download-only snippets. CLI examples include the target
runtime JSON for downloads on a CPU host. Actual execution still requires the
release's matching Ascend environment. Copy controls fall back to text selection
when clipboard access is unavailable.
