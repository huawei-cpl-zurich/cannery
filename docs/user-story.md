# CANNery — End-to-End User Story

Status: proposed user experience and API contract · 2026-11-09

The scenario follows a developer who **registers a new PyTorch operator once**, then submits multiple optimized implementations without resubmitting the specification. A second user discovers a validated published implementation and uses it from Python through the Hugging Face kernel runtime.

All CANNery commands, REST endpoints, JSON outputs and repository identifiers below are **proposed interface examples**; they are not yet implemented. Existing Hugging Face client APIs are identified separately.

See [Architecture](architecture.md) for where files land, what gets copied, and worker execution.

## 1. The first author registers `team/fa_xxx`

Alice has a PyTorch implementation of an attention variant. She wants CANNery to benchmark optimized Ascend implementations against the **same frozen reference and cases**.

Her source tree:

```text
fa_xxx/
  cannery.yaml
  reference.py
  cases.py
  requirements.lock
```

The specification YAML establishes the canonical contract:

```yaml
schema_version: 1
kernel:
  namespace: team
  name: fa_xxx
  description: FlashAttention variant
spec:
  version: 1
  reference:
    file: reference.py
    function: reference
  cases:
    file: cases.py
    function: generate_cases
  interface:
    inputs: [q, k, v]
    outputs: [result]
  validation:
    metric: torch_close
    default_rtol: 0.001
    default_atol: 0.0001
    equal_nan: false
```

The reference and case generator are ordinary Python using PyTorch. The test suite is **user-defined**, not drawn from CANNBench.

### 1.1 Via the website

1. Sign in and select **Register New Kernel**.
2. Enter `team/fa_xxx`; select **Upload specification folder** or **Import from Git**.
3. For Git, enter the repository URL, a **specific full commit SHA**, and `specs/fa_xxx/`. The server freezes the files from that commit.
4. The preview shows `cannery.yaml`, `reference.py`, `cases.py`, the default tolerance and input/output signature.
5. Select **Register**. The UI returns `team/fa_xxx@spec-v1`, a content digest, provenance, and a read-only spec page.
6. Future changes to reference, cases, or validation semantics require **spec-v2**. The author can update descriptive metadata without changing the spec.

At this point the operator exists in the catalog, even if no optimized implementation has been published.

### 1.2 Via the CLI

```bash
# From a local folder
cannery register ./fa_xxx

# Or from a pinned Git repository
cannery register \
  --repo https://git.example.org/team/kernels.git \
  --commit 0123456789abcdef0123456789abcdef01234567 \
  --path specs/fa_xxx/
```

Illustrative output:

```text
✓ Specification validated
✓ Content snapshot uploaded
Kernel       team/fa_xxx
Spec         spec-v1 (frozen)
Digest       sha256:5fe9...
Source       git:0123456789abcdef0123456789abcdef01234567
Status       Registered
```

### 1.3 Via REST

For Git intake, the client submits references and the server performs a restricted snapshot:

```bash
curl -X POST https://cannery.example.internal/api/v1/kernels \
  -H "Authorization: Bearer $CANNERY_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "namespace": "team",
    "name": "fa_xxx",
    "source": {
      "type": "git",
      "url": "https://git.example.org/team/kernels.git",
      "commit": "0123456789abcdef0123456789abcdef01234567",
      "path": "specs/fa_xxx/"
    }
  }'
```

Illustrative response:

```json
{
  "kernel": "team/fa_xxx",
  "spec": "spec-v1",
  "spec_digest": "sha256:5fe9...",
  "status": "registered",
  "immutable": true
}
```

For local uploads, CANNery should provide an upload-init endpoint and a short-lived upload URL. After upload, a finalize operation computes the digest and creates the immutable spec. This avoids pushing large archives through the API worker.

### 1.4 Where registration data lands

- Git repo or browser upload → isolated intake scratch (**temporary**).
- Validated spec files, YAML and dependency lock → immutable bundle in **private HF storage** (**canonical bytes**).
- Kernel ID, spec version, digest, Git provenance and pinned HF URI → **PostgreSQL** (**canonical index**).
- API/browser keeps only metadata, not a duplicate permanent source checkout.

## 2. Alice submits her first optimized implementation

Alice implements the frozen attention contract using CATLASS DSL on A5. She does not include the PyTorch spec again.

```text
optimized-fa/
  submission.yaml
  kernel.py
```

```yaml
schema_version: 1
kernel: team/fa_xxx
spec_version: 1
implementation:
  device: a5
  language: catlass
  entrypoint: kernel.py:forward
validation:
  requested_rtol: 0.001
  requested_atol: 0.0001
```

### 2.1 Via the website

1. Open **team/fa_xxx** and select **Submit Implementation**.
2. The UI shows the frozen `spec-v1`, PyTorch reference and cases as **read-only**.
3. Select device **A5** and language **CATLASS**.
4. Choose **Upload source** or **Git repo**, entering URL, full commit SHA and source subdirectory.
5. Provide `kernel.py:forward`, requested `rtol` and `atol`, and submit.
6. **My Submissions** shows the new revision `r001`: queued → compiling → correctness → benchmarking → complete/failed. It shows an empirical **ETA range** or “insufficient history,” not worker capacity or host IDs.
7. Compilation and correctness logs are available. **No final performance score appears until correctness and all mandatory benchmark cases have completed.**

### 2.2 Via the CLI

```bash
# A local source snapshot
cannery submit team/fa_xxx ./optimized-fa \
  --spec-version 1 --device a5 --lang catlass

# Or a revision pinned to a Git commit
cannery submit team/fa_xxx \
  --spec-version 1 --device a5 --lang catlass \
  --repo https://git.example.org/team/optimized-kernels.git \
  --commit fedcba9876543210fedcba9876543210fedcba98 \
  --path attention/fa_xxx/
```

Illustrative output:

```text
Submission    team/fa_xxx / r001
Spec          spec-v1 (sha256:5fe9...)
Source        sha256:9e13...
Status        queued
Est. test     8–15 min (based on historical A5 CATLASS jobs)
Track         A5 / CATLASS
```

### 2.3 Via REST

```bash
curl -X POST https://cannery.example.internal/api/v1/kernels/team/fa_xxx/submissions \
  -H "Authorization: Bearer $CANNERY_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "spec_version": 1,
    "device": "a5",
    "language": "catlass",
    "entrypoint": "kernel.py:forward",
    "source": {
      "type": "git",
      "url": "https://git.example.org/team/optimized-kernels.git",
      "commit": "fedcba9876543210fedcba9876543210fedcba98",
      "path": "attention/fa_xxx/"
    },
    "requested_tolerance": {"rtol": 0.001, "atol": 0.0001}
  }'
```

Response:

```json
{
  "submission_id": "sub_001",
  "kernel": "team/fa_xxx",
  "spec_version": 1,
  "source_digest": "sha256:9e13...",
  "status": "queued",
  "job_id": "job_001"
}
```

Retrieve progress:

```bash
cannery jobs show job_001
curl -H "Authorization: Bearer $CANNERY_TOKEN" \
  https://cannery.example.internal/api/v1/jobs/job_001
```

## 3. The platform validates and benchmarks the implementation

The scheduler finds an A5 worker supporting CATLASS and matching toolchains. The worker coordinator downloads the pinned spec and candidate snapshot to an isolated scratch directory, builds/loads the candidate, and runs the canonical PyTorch reference and candidate over **all mandatory cases**.

The worker:
1. Acquires an exclusive device lease for timing.
2. Compiles and runs correctness tests for the requested/versioned tolerance profiles.
3. Measures median/p95 execution latency **in µs** after warmup and appropriate NPU synchronization.
4. Computes each case's speedup against its pinned baseline and the **geometric mean of speedup ratios** across all mandatory cases.
5. Uploads raw timings, correctness detail, logs, binary and build manifest into private HF storage.
6. Writes result summaries and immutable artifact pointers to PostgreSQL through the trusted coordinator/API; releases the device.

Example completed report (illustrative):

```text
team/fa_xxx / r001       A5 / CATLASS
Spec                     spec-v1
Validation (rtol=1e-3, atol=1e-4)  PASS (4/4 mandatory cases)

Case                  Baseline     Candidate    Speedup
batch1_seq512         28.4 µs      19.2 µs       1.48×
batch4_seq1024        94.5 µs      52.1 µs       1.81×
batch8_seq4096        680.0 µs     310.0 µs      2.19×
batch16_seq4096       1290.0 µs    550.0 µs      2.35×

Geomean speedup: 1.91×
Status: Validated — private (not globally published)
```

A failed correctness result retains error logs but never produces a final leaderboard-eligible speedup. A separate A3 submission goes through independent benchmark cases/environment and is ranked only against A3 results.

**Data movement:** private HF spec and source → local worker scratch (**copies**); worker-generated binary/results → private HF (**canonical artifacts**); job status, hashes, summaries, scores → PostgreSQL (**canonical searchable records**); Redis transports the job ID only. Worker scratch is deleted or retained briefly according to debugging policy.

## 4. Alice iterates without changing the spec

Alice optimizes the implementation and pushes a new Git commit:

```bash
cannery submit team/fa_xxx \
  --spec-version 1 --device a5 --lang triton \
  --repo https://git.example.org/team/optimized-kernels.git \
  --commit 1111111111111111111111111111111111111111 \
  --path attention/fa_xxx/
```

This creates `r002`, another **immutable** submission pointing at the **same frozen spec-v1**. All previous results remain reproducible.

If Alice discovers that the reference itself is wrong, she must register `spec-v2`; it starts a **separate comparability group** rather than rewriting old results.

## 5. Alice publishes a validated result

She opens the validated submission and clicks **Publish** in My Submissions, or uses:

```bash
cannery publish team/fa_xxx --submission sub_001
```

Proposed REST:

```bash
curl -X POST https://cannery.example.internal/api/v1/submissions/sub_001/publish \
  -H "Authorization: Bearer $CANNERY_TOKEN"
```

Publication performs a second artifact step: it builds/verifies a **Hugging Face kernels-compatible release layout** and pins the resulting private HF repository revision. CANNery records the mapping:

```json
{
  "cannery_submission": "sub_001",
  "kernel": "team/fa_xxx",
  "spec_version": 1,
  "device": "a5",
  "language": "catlass",
  "huggingface": {
    "repo_id": "cannery-org/fa-xxx-a5-release",
    "repo_type": "kernel",
    "revision": "<immutable-hf-commit-sha>"
  },
  "published": true
}
```

The HF name, repository type, and compatibility manifest above are **target release conventions**, subject to repository creation permission and `cannery-kernels` integration. A private ordinary HF model repo is enough for archived source distribution but does **not** automatically guarantee compatibility with `kernels.get_kernel`.

The public-facing CANNery Global Best view shows only **fully validated, explicitly published and currently eligible** results. A user filtering to stricter `rtol/atol` sees only implementations proven to pass that profile; unseen profiles trigger revalidation rather than guessing from summary errors.

For A5, the performance history draws one cumulative **best-so-far** step line; the event points are colored by the language that produced each new global record. An A3 user gets a different chart and rankings.

## 6. Bob discovers a kernel

Bob opens **Global Best**, searches for `fa_xxx`, selects **A5** and a tolerance profile. He sees:
- fully published and validated A5 implementations ranked by geomean speedup;
- per-case median latency in µs, baseline and achieved error;
- the frozen PyTorch reference, case suite and author/Git provenance;
- an exact **HF repository ID + immutable revision** for every downloadable release;
- source and compatible binary download options.

Bob can use CANNery's proposed CLI to resolve a compatible artifact:

```bash
cannery info team/fa_xxx
cannery get team/fa_xxx --device a5 --lang catlass
cannery get team/fa_xxx --device a5 --source

# Explicitly choose a variant without local probing
cannery get team/fa_xxx --device a5 --lang catlass --no-detect
```

Automatic selection probes local A3/A5, CANN and torch_npu versions, Python ABI and available language runtimes. It downloads by immutable revision and verifies artifact hashes. If a binary is incompatible, it can offer source, but it should not silently compile and execute untrusted remote code.

## 7. Bob uses the published kernel through Hugging Face

### 7.1 Direct Hugging Face `kernels` API

Hugging Face's existing Python API looks like this:

```python
from kernels import get_kernel

# A pinned release; exact repo/revision comes from CANNery's published manifest.
# Requires Ascend support from cannery-kernels (planned, not yet implemented).
module = get_kernel(
    "cannery-org/fa-xxx-a5-release",
    revision="<immutable-hf-commit-sha>",
    backend="cann",
    trust_remote_code=["cannery-org/fa-xxx-a5-release"],
)

# Target callable exported by the release's kernel module:
out = module.forward(q, k, v)
```

**Important:** `get_kernel(repo_id, revision=..., backend=..., trust_remote_code=...)` is an existing Hugging Face API, but `backend="cann"` plus the CATLASS/Triton/Ascend C release loading path are **CANNery fork features to develop/test**. The `module.forward` export is part of our proposed packaging contract, not guaranteed by upstream. Repository trust is intentionally restricted to the selected release, and remote code must remain subject to the organization's security policy.

Upstream API reference: https://huggingface.co/docs/kernels/en/api/kernels

### 7.2 Download source directly with `huggingface_hub`

A user who only wants to inspect or locally build source can use Hugging Face's **existing** SDK after obtaining the published source repository/revision from CANNery:

```python
from huggingface_hub import snapshot_download

source_path = snapshot_download(
    repo_id="cannery-org/fa-xxx-sources",
    repo_type="model",  # PoC source archive repository type
    revision="<immutable-source-hf-commit-sha>",
    token=True,         # use logged-in user's HF identity
)
print(source_path)
```

`snapshot_download` is supported today and caches files locally. The source archive's exact paths and manifest are CANNery conventions. There is **no execution** from merely downloading the snapshot.

HF download documentation: https://huggingface.co/docs/huggingface_hub/en/guides/download

### 7.3 CANNery convenience wrapper (proposed)

A small Python API in CANNery could query its leaderboard and select the best **validated and published** compatible release given the user's tolerance:

```python
from cannery_client import resolve_kernel  # proposed API

release = resolve_kernel(
    "team/fa_xxx",
    device="auto",
    rtol=1e-3,
    atol=1e-4,
)
print(release.hf_repo_id, release.hf_revision)

# The runtime can then use the pinned HF release.
from kernels import get_kernel
module = get_kernel(
    release.hf_repo_id,
    revision=release.hf_revision,
    backend="cann",  # requires cannery-kernels Ascend extension
    trust_remote_code=[release.hf_repo_id],
)
out = module.forward(q, k, v)
```

This wrapper is a **proposed CANNery layer**, not an existing PyPI API. It does not replace Hugging Face artifact storage or kernel loading.

## 8. Composing kernels

Bob retrieves three published components (an official validated kernel and two unofficial/author-namespace kernels), each pinned to its own revision. He loads each as an independent module, preserving provenance and compatibility, and combines their callable interfaces in a PyTorch operator:

```python
from kernels import get_kernel

# Example repo IDs/revisions only; requires Ascend-enabled loader.
norm = get_kernel("cannery-org/rms-norm-a5", revision="<norm-sha>",
                  backend="cann", trust_remote_code=["cannery-org/rms-norm-a5"])
attn = get_kernel("cannery-org/fa-xxx-a5-release", revision="<attn-sha>",
                  backend="cann", trust_remote_code=["cannery-org/fa-xxx-a5-release"])
proj = get_kernel("cannery-org/linear-a5", revision="<proj-sha>",
                  backend="cann", trust_remote_code=["cannery-org/linear-a5"])

def composed_operator(x, q, k, v, weight):
    x = norm.forward(x)
    attention = attn.forward(q, k, v)
    return proj.forward(x, attention, weight)
```

These `forward` signatures are **illustrative**. Actual composition must obey the frozen per-operator tensor/shape contracts and supported entrypoints. A private release still requires permission; publishing in CANNery doesn't automatically make HF bytes public.

## 9. Acceptance criteria for the user journey

- A new kernel requires a frozen YAML + PyTorch reference/cases; later implementations only reference the canonical spec.
- Both UI and programmatic flows support local source or pinned Git URL/commit/path.
- The server stores immutable source/spec snapshots independent of the original Git repository.
- Users see validation phase, useful failure details and estimated completion time, not infrastructure details.
- Each case's latency is in **µs** with geometric-mean **speedup** across all required cases.
- Incorrect/unvalidated/unpublished candidates never appear on Global Best.
- Tolerance filters use **proven** validation outcomes.
- A3 and A5 leaderboard/historical comparisons are isolated.
- A published release resolves to an **immutable HF revision**, with loadable Ascend kernel artifacts after the fork extension is implemented.
- A consumer can inspect/download source through the HF SDK and load/execute the published binary/source module through the HF kernel runtime with explicit trust policy.

## 10. Data-flow summary

```text
REGISTER:
Git @ SHA / browser upload
  → isolated intake temp
  → private HF spec snapshot (canonical bytes)
  → PostgreSQL spec digest + pinned HF URI (canonical index)

SUBMIT:
Git @ SHA / local source
  → isolated intake temp
  → private HF source snapshot
  → PostgreSQL submission row + job → Redis ID
  → A3/A5 worker downloads spec and source to scratch
  → builds + PyTorch correctness + device benchmark
  → private HF binary and detailed run bundles
  → PostgreSQL validated case results, score and artifact pointers

PUBLISH:
owner approval → loader-compatible HF release, pinned revision
  → PostgreSQL published mapping and leaderboard eligibility

CONSUME:
CANNery catalog/SDK resolution
  → pinned HF repo/revision
  → Hugging Face SDK cache on consumer host
  → cannery-kernels loader
  → CANN / torch_npu / NPU execution
```
