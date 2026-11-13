# CANNery — End-to-End User Story

Current A5 contracts include [Ascend C admission](binary-admission.md),
[CATLASS admission](catlass-admission.md), [seeded numerical validation](numerical-worker.md)
and [private performance reports](benchmark-worker.md). Artifacts are self-hosted;
no actual Hugging Face service is required. [Publication and Global Best](publication.md)
are implemented for A5; A3 remains planned.


Status: proposed PoC journey · 2026-11-09

All **user-authored source code stays in Git**. The platform only accepts pinned Git commits and source paths. During admission it creates a reproducible **generated binary** with one of the supported language adapters. The binary and benchmark evidence are stored in CANNery's self-hosted private artifact backend; source never goes into a artifact source archive.

Commands and REST responses below are **proposed CANNery APIs**, not implemented endpoints. Private artifact APIs are documented separately.

## 1. Alice registers a new operator using the website

Alice has a repository at `https://git.example.org/team/kernel-lab.git` with:

```text
kernel-lab/
  specs/fa_xxx/
    cannery.yaml
    reference.py
    cases.py
    requirements.lock
  implementations/fa_xxx/
    submission.yaml
    kernel.py
```

The frozen spec `specs/fa_xxx/cannery.yaml` is:

```yaml
schema_version: 1
kernel:
  namespace: team
  name: fa_xxx
spec:
  version: 1
  reference: {file: reference.py, function: reference}
  cases: {file: cases.py, function: generate_cases}
  interface:
    inputs: [q, k, v]
    outputs: [result]
  validation:
    metric: torch_close
    default_rtol: 0.001
    default_atol: 0.0001
    equal_nan: false
```

She signs in, selects **Register Kernel**, chooses **Git repository**, and enters the repo URL, full commit SHA and `specs/fa_xxx/`. The UI previews the YAML, PyTorch reference and generated cases, and shows that publication freezes the spec. She presses **Register**.

CANNery fetches the commit in a restricted Git intake service, imports only the selected spec directory and declared dependencies into an **internal per-kernel Git snapshot repository**, verifies the tree and files, then registers `team/fa_xxx@spec-v1` in PostgreSQL with the full SHA and content digests. It **does not upload a source snapshot to Hugging Face**. The author can register a new `spec-v2` later; `spec-v1` is never edited.

### Equivalent CLI

```bash
cannery register \
  --repo https://git.example.org/team/kernel-lab.git \
  --commit 0123456789abcdef0123456789abcdef01234567 \
  --path specs/fa_xxx/
```

Mock output:

```text
Registered: team/fa_xxx@spec-v1
Git:        0123456789abcdef0123456789abcdef01234567
Spec hash:  sha256:5fe9...
Storage:    internal Git snapshot commit (selected files only)
State:      frozen
```

### Equivalent REST

```bash
curl -X POST https://cannery.example.internal/api/v1/kernels \
  -H "Authorization: Bearer $CANNERY_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "namespace": "team",
    "name": "fa_xxx",
    "source": {
      "type": "git",
      "url": "https://git.example.org/team/kernel-lab.git",
      "commit": "0123456789abcdef0123456789abcdef01234567",
      "path": "specs/fa_xxx/"
    }
  }'
```

No upload-source endpoint exists.

## 2. Alice submits an implementation to the frozen spec

Alice adds an A5 CATLASS implementation to Git. The submission manifest lives beside her kernel and makes the **build reproducible**:

```yaml
schema_version: 1
kernel: team/fa_xxx
spec_version: 1
implementation:
  device: a5
  language: catlass
  entrypoint: kernel.py:forward
  build:
    adapter: catlass
    inputs: [kernel.py]
    artifact: fa_xxx_a5
    options:
      optimization: release
validation:
  requested_rtol: 0.001
  requested_atol: 0.0001
```

The PoC accepts `ascendc` and `catlass`. Triton and PyPTO adapters are planned later; selecting those languages during the PoC yields **unsupported adapter**, not an unbuildable submission.

In the UI she opens `team/fa_xxx` → **Submit Implementation**, selects **Git**, enters repository, full commit and `implementations/fa_xxx/`, and selects A5 / CATLASS. She sees a **Build contract verified** preflight when the YAML is valid. The UI reports an estimated testing time range. After pressing Submit, **My Submissions** shows `r001`: queued → building → validating → benchmarking → complete/failed.

No user-supplied prebuilt binary is accepted as the primary submission. The worker creates it during admission.

### CLI and REST

```bash
cannery submit team/fa_xxx \
  --spec-version 1 --device a5 --lang catlass \
  --repo https://git.example.org/team/kernel-lab.git \
  --commit fedcba9876543210fedcba9876543210fedcba98 \
  --path implementations/fa_xxx/
```

```bash
curl -X POST https://cannery.example.internal/api/v1/kernels/team/fa_xxx/submissions \
  -H "Authorization: Bearer $CANNERY_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "spec_version": 1,
    "source": {
      "type": "git",
      "url": "https://git.example.org/team/kernel-lab.git",
      "commit": "fedcba9876543210fedcba9876543210fedcba98",
      "path": "implementations/fa_xxx/"
    },
    "device": "a5",
    "language": "catlass"
  }'
```

Response example:

```json
{
  "submission_id": "sub_001",
  "parent_submission_id": null,
  "source_commit": "fedcba9876543210fedcba9876543210fedcba98",
  "source_tree_digest": "sha256:9e13...",
  "build_adapter": "catlass",
  "state": "queued_for_build",
  "job_id": "job_001"
}
```

## 3. Binary generation and validation

CANNery's scheduler matches the submission to a worker with A5 + CATLASS build/execute capabilities. The trusted coordinator checks out frozen spec-v1 and the candidate's Git commit from the internal Git snapshot repository, inside an isolated environment.

1. The CATLASS build adapter compiles the declared supported specializations into **binary candidate(s)**. For frameworks whose compilation is JIT-dependent, the adapter must proactively compile/export each supported specialization using the real framework API. If it cannot export a loadable artifact, the submission cannot be advertised as a prebuilt binary.
2. A build manifest captures source Git SHA/tree digest, adapter version, CANN, torch_npu, target device, ABI, supported shapes/dtypes and output checksums.
3. The generated binary and build report go to **private content-addressed staging**. CANNery stores only their pinned URI/hash in PostgreSQL.
4. Correctness runs the frozen PyTorch reference and compiled candidate on the same mandatory cases. It evaluates stored tolerance profiles. Failures retain private logs and never get a valid public performance score.
5. After correctness passes, the worker times kernel executions on A5, reports median/p95 in **µs** by case and geomean of baseline-to-candidate speedups.

Illustrative completed report:

```text
team/fa_xxx / r001       A5 / CATLASS
Build                    SUCCESS (adapter catlass)
Binary                   hf://cannery-org/fa-xxx-builds@<pinned-revision>
Correctness              PASS (4/4 cases; rtol=1e-3, atol=1e-4)

Case                  Baseline       Candidate       Speedup
batch1_seq512         28.4 µs        19.2 µs         1.48x
batch4_seq1024        94.5 µs        52.1 µs         1.81x
batch8_seq4096        680.0 µs       310.0 µs        2.19x
batch16_seq4096       1290.0 µs      550.0 µs        2.35x

Geomean speedup        1.91x
Visibility             PRIVATE — validated, not published
```

These figures are examples, not actual benchmarks. During correctness or benchmarking the UI displays a **provisional stage**, not the final leaderboard table.

Data movement:

```text
External Git --fetch--> Per-kernel internal Git snapshots (canonical retained source)
                               |
                               | checkout exact SHA + path
                               v
                          Isolated worker
                     build -> test -> benchmark
                               |
             +-----------------+------------------+
             v                                    v
 Private artifact service                       PostgreSQL
 generated binary + build and raw logs      source refs + states + scores
```

## 4. Alice makes a second submission by changing only the commit

She improves her Git code and commits it. In **My Submissions**, she clicks **New Revision**. The repository, path, frozen spec, device, language, build adapter and entrypoint are inherited. She only changes **Commit SHA**.

```bash
cannery revise sub_001 \
  --commit 1111111111111111111111111111111111111111
```

Equivalent proposed REST:

```bash
curl -X POST https://cannery.example.internal/api/v1/submissions/sub_001/revisions \
  -H "Authorization: Bearer $CANNERY_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"commit": "1111111111111111111111111111111111111111"}'
```

This produces `r002`, a new frozen Git source binding and a new build, binary, correctness report and benchmark. Nothing in `r001` is mutated.

An update to `reference.py`, `cases.py` or `cannery.yaml` belongs to **Register spec-v2**, not a revision of an implementation submission.

## 5. Alice publishes a validated binary

She selects a validated revision and chooses **Publish**, or:

```bash
cannery publish team/fa_xxx --submission sub_002
```

CANNery packages the previously generated, validated binaries into a **private runtime-compatible release**, pins its content hash and registers that release in PostgreSQL. No new source upload takes place. If its build cannot be represented by a compatible loadable artifact, publication must fail with a packaging diagnostic rather than publishing an unusable entry.

The global leaderboard includes this implementation only when it is validated **and explicitly published**, and only for tolerance profiles it has passed. A5 and A3 results have separate histories; each history contains one overall best-so-far step line with record points colored by language.

## 6. Bob consumes the kernel

Bob visits **Global Best**, selects `team/fa_xxx`, **A5** and his tolerance. He sees a published validated release, case-level µs, geomean speedup, errors, and two distinct links:

- **Source (Git):** the external repository + full SHA + path (or authorized internal snapshot repository URL). No source files are served from Hugging Face.
- **Binary (private storage):** the exact private blob reference and immutable artifact hash with validated binary and callable packaging.

CLI examples:

```bash
cannery info team/fa_xxx
cannery source team/fa_xxx --revision r002
cannery get team/fa_xxx --device a5 --lang catlass
```

### Load the private admitted package

M3 provides `cannery_kernels.load_cannery_package(package_bytes, pinned_sha256,
trust_remote_code=True)` from the standalone `cannery-runtime` distribution.
The caller obtains bytes through authorized private storage, verifies the pinned
reference and explicitly trusts execution of the native binary. Its initial
`forward(x)` contract is float32 unary A5; general release resolution and
publication remain future work. No actual Hugging Face endpoint is used.

### Inspect the original source

```bash
git clone https://git.example.org/team/kernel-lab.git
cd kernel-lab
git checkout fedcba9876543210fedcba9876543210fedcba98
ls implementations/fa_xxx/
```

Source access is subject to Git permissions, while binary download is subject to private storage permissions. The admitted source tree is retained as an internal Git snapshot commit if the developer's remote disappears. Private Git source access need not be granted to binary consumers.

## 7. PoC completion conditions

- **Git-only** inputs for spec and implementations: URL, immutable commit, path; no local source archive ingestion.
- Saved Git binding supports **commit-only revisions**, each producing a separate immutable submission.
- Frozen PyTorch YAML/reference/cases live in internal Git snapshots and PostgreSQL digests, never mutable in place.
- Each accepted implementation declares a supported, reproducible build plan; build runs automatically at admission.
- PoC adapters support **Ascend C** and **CATLASS DSL** only; Triton and PyPTO are future integrations.
- Generated binary plus build/environment manifest is stored as a pinned private content-addressed artifact.
- Validation and benchmark use built candidate, not an author's arbitrary supplied binary.
- Per-case latency, accuracy and geomean are recorded; provisional/failed results do not enter Global Best.
- A consumer can follow Git to inspect source and use the pinned private `cannery-runtime` interface to load a published A3/A5 binary once loader support is implemented.
