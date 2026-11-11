# CATLASS binary admission contract

M4 binary admission is opt-in for A5 with `language: catlass`,
`build.adapter: catlass`, and `build.abi: catlass-unary-f32-v1`:

```yaml
implementation:
  device: a5
  language: catlass
  entrypoint: kernel.py:forward
  build:
    adapter: catlass
    abi: catlass-unary-f32-v1
    inputs: [kernel.py]
    artifact: add_one_a5
    options: {optimization: release}
    specializations:
      - {shape: [17], dtype: float32}
      - {shape: [3, 7], dtype: float32}
      - {shape: [0], dtype: float32}
```

Declare one to four distinct logical shapes, each with one to four nonnegative
integer dimensions and at most 1,048,576 elements. Only float32 and release
compilation are initially supported. Source intake never imports the Python
module. Shape/dtype declarations participate in the frozen source identity.
Older source-only CATLASS manifests and Ascend C manifests retain their hashes.

At least one specialization must be nonempty. Positive element counts are compiled
once each; logical shapes with the same count share the embedded device object.

The unary contract supplies flattened contiguous input/output tensors to the
kernel; the consumer preserves the original logical shape. Declared empty inputs
require no launch. Unsupported shapes and other kernel argument layouts must be
rejected; a lazy JIT fallback does not establish binary admission.

The compiler admits only the pinned A5 static AIV two-pointer ABI, with one vector
block, no mixed Cube/vector execution, debug printing, dynamic UB or scalar
parameters. It embeds the exported device objects in a native AscendCL library.
The source-free schema-2 package contains only `metadata.json` and `kernel.so`.
The offline SDK needs no CATLASS installation or runtime compiler; it rejects
undeclared shapes, other dtypes/devices and a changed ACL context. Keep a loaded
kernel alive while launches are in flight; reload after a device/context reset.

The coordinator compiles author Python only in a device-less, network-isolated,
nonroot container. It verifies the configured compiler/helper fingerprints,
round-trips the resulting package through private storage, and load/call smokes
every declared shape in a fresh restricted A5 runtime container. Success becomes
`awaiting_validation`. Source-only manifests remain `awaiting_build`.
Correctness scoring and benchmarking remain M5.

## Configure the builder

Use the [binary admission setup](binary-admission.md), SDK wheel version 0.2.0
or later, and a pinned image containing CATLASS. Add a trusted `catlass` object
to the worker profile with these required fields:

| Field | Value to record |
|---|---|
| `version` | Exact installed CATLASS version |
| `revision` | Full reviewed CATLASS source revision (40 lowercase hex digits) |
| `execution_sha256` | SHA256 of installed `catlass.execution.__file__` |
| `hivmc_sha256` | SHA256 of the registered runtime’s `hivmc-a5` executable |
| `bitcode_key` | The helper cache’s 16-hex compiler key |
| `manifest_sha256` | SHA256 of the matching helper `manifest.json` |
| `bitcode_sha256` | SHA256 of `meta_op.aiv.c310.bc` |
| `bitcode_root` | Specific absolute host directory containing the keyed helper cache |

Prebuild normal AIV helper bitcode with the exact pinned package’s
`catlass.bc_compile` tool, independently of author submissions. Retain
`<bitcode_root>/<bitcode_key>/{manifest.json,meta_op.aiv.c310.bc}` read-only and
record the hashes. The validated toolchain uses CATLASS revision
`be0d1c5752d9381254eda336fcc135755653024a`, architecture 3510/c310 and CANN 9.3.0;
other toolchains need separate export/launch validation. The cache key alone
is insufficient: the driver verifies both helper file hashes before author import.

Declare `CATLASS_DSL_ASCENDNPU_IR_ROOT` and any framework/compiler search paths
in the registered named runtime. The supervisor resolves that environment and
mounts the IR installation and helper cache read-only only during CATLASS
compilation, alongside a bounded writable private cache. Runtime jobs receive
neither mount. A missing or changed toolchain fails admission; it never falls
back to JIT. Drain active builds before updating the image, profile or SDK wheel.

## Try the public example

Open `/submit-implementation?example=helloworld-catlass` after registering
`team/helloworld@spec-v1`, or run:

```bash
cannery submit team/helloworld --spec-version 1 --device a5 --lang catlass \
  --repo https://github.com/huawei-cpl-zurich/helloworld-kernel.git \
  --commit f840604437aec7f42d66bf281680a39ea2a03f04 \
  --path candidates/helloworld_catlass --url http://localhost:8095 \
  --wait-build --timeout 900
```

Keep the exact repository allowlisted and use a full immutable commit SHA.
The example declares `[17]`, `[3, 7]` and `[0]`; it makes no performance claim.
