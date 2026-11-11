# CATLASS binary admission contract

The initial M4 declaration is opt-in for A5 with `language: catlass`,
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

The unary contract supplies flattened contiguous input/output tensors to the
kernel; the consumer preserves the original logical shape. Declared empty inputs
require no launch. Unsupported shapes and other kernel argument layouts must be
rejected; a lazy JIT fallback does not establish binary admission.

This slice adds declaration validation only. Binary compilation, packaging and
runtime dispatch are separate M4 integration work; until those are enabled, an
accepted source remains `awaiting_build`. Correctness scoring and benchmarking
remain M5.
