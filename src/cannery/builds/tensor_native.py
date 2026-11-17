"""Generate source-free tensor-v2 launches; initialization happens before forward."""

import json
import math
import re
import struct
from typing import Any

from cannery.builds.ascendc import BuildError
from cannery.builds.catlass import MAX_OBJECT_BYTES, _native_object
from cannery.builds.catlass_tensor import StaticTensorExport

DTYPES = {
    name: index + 1
    for index, name in enumerate(
        ["float32", "float16", "bfloat16", "int8", "int32", "int64", "bool"]
    )
}
WIDTHS = {"float32": 4, "float16": 2, "bfloat16": 2, "int8": 1, "int32": 4, "int64": 8, "bool": 1}


def _conditions(signature: dict[str, Any]) -> str:
    inputs, outputs, scalars = (signature[key] for key in ("inputs", "outputs", "scalars"))
    if not 1 <= len(inputs) <= 32 or not 1 <= len(outputs) <= 32 or len(scalars) > 32:
        raise ValueError("Invalid argument counts")
    conditions = [f"ni=={len(inputs)}", f"no=={len(outputs)}", f"ns=={len(scalars)}"]
    size = 0
    for direction, tensors in [("in", inputs), ("out", outputs)]:
        if len({t["name"] for t in tensors}) != len(tensors):
            raise ValueError("Duplicate tensor name")
        for index, tensor in enumerate(tensors):
            shape, dtype = tensor["shape"], tensor["dtype"]
            if (
                type(shape) is not list
                or not 1 <= len(shape) <= 4
                or any(type(d) is not int or not 0 <= d <= 1048576 for d in shape)
                or math.prod(shape) > 1048576
            ):
                raise ValueError("Invalid tensor shape")
            size += math.prod(shape) * WIDTHS[dtype]
            prefix = f"{direction}[{index}]"
            conditions += [f"{prefix}.rank=={len(shape)}", f"{prefix}.dtype=={DTYPES[dtype]}"]
            if math.prod(shape):
                conditions.append(f"{prefix}.data!=nullptr")
            strides, stride = [0] * 4, 1
            for axis in range(len(shape) - 1, -1, -1):
                strides[axis] = stride
                stride *= max(shape[axis], 1)
            for axis in range(4):
                extent = shape[axis] if axis < len(shape) else 0
                conditions += [
                    f"{prefix}.sizes[{axis}]==UINT64_C({extent})",
                    f"{prefix}.strides[{axis}]==UINT64_C({strides[axis]})",
                ]
            if direction == "out" and tensor["alias"] is not None:
                original = next(i for i, t in enumerate(inputs) if t["name"] == tensor["alias"])
                if any(inputs[original][key] != tensor[key] for key in ("shape", "dtype")):
                    raise ValueError("Invalid mutable alias")
                conditions.append(f"{prefix}.data==in[{original}].data")
    if size > 32 * 1024 * 1024:
        raise ValueError("Tensor byte budget exceeded")
    if len({s["name"] for s in scalars} | {t["name"] for t in inputs}) != len(scalars) + len(
        inputs
    ):
        raise ValueError("Duplicate input/scalar name")
    for index, scalar in enumerate(scalars):
        kind, value = scalar["kind"], scalar["value"]
        if kind == "bool" and type(value) is bool:
            code, bits = 1, int(value)
        elif kind == "int64" and type(value) is int and -(2**63) <= value < 2**63:
            code, bits = 2, value % 2**64
        elif kind == "float64" and type(value) is float and math.isfinite(value):
            code, bits = 3, struct.unpack("<Q", struct.pack("<d", value))[0]
        else:
            raise ValueError("Invalid static scalar")
        conditions += [
            f"scalar[{index}].kind=={code}",
            f"scalar[{index}].reserved==0",
            f"scalar[{index}].bits==UINT64_C({bits})",
        ]
    return " && ".join(conditions)


def native_tensor_wrapper(
    variants: list[tuple[dict[str, Any], StaticTensorExport]],
    *,
    triton_grids: list[tuple[int, int, int]] | None = None,
) -> str:
    """Use the tensor_abi.h from the pinned runtime wheel when compiling this source."""
    try:
        if (
            not 1 <= len(variants) <= 32
            or sum(len(v.binary) for _, v in variants) > MAX_OBJECT_BYTES
        ):
            raise ValueError("Native image budget exceeded")
        if triton_grids is not None and (
            type(triton_grids) is not list or len(triton_grids) != len(variants)
        ):
            raise ValueError("Invalid static Triton grids")
        objects, records, launches, domains = [], [], [], set()
        interface = None
        for index, (signature, export) in enumerate(variants):
            if (
                not _native_object(export.binary)
                or export.mode not in ("aic", "aiv")
                or type(export.block_num) is not int
                or not 1 <= export.block_num <= 65535
                or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", export.entrypoint) is None
                or not 1 <= len(export.bindings) <= 64
            ):
                raise ValueError("Invalid verified export")
            domain = json.dumps({k: signature[k] for k in ("inputs", "scalars")}, sort_keys=True)
            if domain in domains:
                raise ValueError("Duplicate native dispatch domain")
            domains.add(domain)
            guard = _conditions(signature)
            current = (
                tuple(t["name"] for t in signature["inputs"]),
                tuple(t["name"] for t in signature["outputs"]),
                tuple((s["name"], s["kind"]) for s in signature["scalars"]),
            )
            if interface is not None and current != interface:
                raise ValueError("Native variants must share the ordered interface")
            interface = current
            arguments = []
            for direction, position in export.bindings:
                if (
                    direction not in ("input", "output")
                    or type(position) is not int
                    or not 0 <= position < len(signature[direction + "s"])
                ):
                    raise ValueError("Invalid native pointer binding")
                arguments.append(f"{'in' if direction == 'input' else 'out'}[{position}].data")
            objects.append(
                f"alignas(64) static const unsigned char image_{index}[]={{"
                + ",".join(map(str, export.binary))
                + "};"
            )
            magic = "VECTOR_CORE" if export.mode == "aiv" else "AICORE"
            records.append(
                f"{{image_{index},sizeof(image_{index}),"
                + json.dumps(export.entrypoint)
                + f",ACL_RT_BINARY_MAGIC_ELF_{magic}}}"
            )
            argument_struct = f"void* args[]={{ {','.join(arguments)} }};"
            if triton_grids is not None:
                grid = triton_grids[index]
                if (
                    type(grid) is not tuple
                    or len(grid) != 3
                    or any(type(value) is not int or not 1 <= value <= 65535 for value in grid)
                    or math.prod(grid) != export.block_num
                ):
                    raise ValueError("Triton grid and native block count differ")
                # Pinned Ascend950 non-SIMT launcher ABI: sync/workspace, pointers,
                # three int32 grid extents, then an 8-byte aligned debug pointer.
                argument_struct = (
                    f"struct {{ void* sync; void* workspace; void* pointers[{len(arguments)}]; "
                    "int32_t gx,gy,gz; void* debug; } args={nullptr,nullptr,{"
                    + ",".join(arguments)
                    + "},"
                    + ",".join(map(str, grid))
                    + ",nullptr};"
                )
            launches.append(f"""if ({guard}) {{
    {argument_struct}
    return aclrtLaunchKernelWithHostArgs(variants[{index}].function,{export.block_num},
        stream,nullptr,{"&args" if triton_grids is not None else "args"},sizeof(args),nullptr,0);
  }}""")
        return (
            '#include <acl/acl.h>\n#include <mutex>\n#include "tensor_abi.h"\n'
            + "\n".join(objects)
            + "\n"
            + "struct Variant { const unsigned char* image; size_t size; const char* symbol;\n"
            "uint32_t magic; aclrtBinHandle binary{}; aclrtFuncHandle function{}; };\n"
            "static Variant variants[]={"
            + ",".join(records)
            + "};\n"
            + PREPARE
            + """
extern "C" int32_t cannery_launch_tensor_v2(void* stream,
    uint32_t ni,const CanneryTensorV2* in,uint32_t no,const CanneryTensorV2* out,
    uint32_t ns,const CanneryScalarV2* scalar) {
  std::lock_guard<std::mutex> guard(launch_mutex);
  if(!ready || !in || !out || (ns && !scalar)) return -11;
  aclrtContext current=nullptr;
  auto status=aclrtGetCurrentContext(&current);
  if(status) return status;
  if(current!=bound_context) return -12;
"""
            + "\n".join(launches)
            + "\nreturn -13;\n}\n"
        )
    except (
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        StopIteration,
        OverflowError,
        struct.error,
    ):
        raise BuildError("Invalid static native tensor variants") from None


PREPARE = r"""
static std::mutex launch_mutex;
static aclrtContext bound_context=nullptr;
static bool ready=false;
static int32_t cleanup_error=0;
static int32_t release_partial() {
  int32_t failure=0;
  for(auto& v:variants) {
    v.function=nullptr;
    if(v.binary) {
      auto status=aclrtBinaryUnLoad(v.binary);
      if(status) {if(!failure) failure=status; continue;}
      v.binary=nullptr;
    }
  }
  return failure;
}
extern "C" int32_t cannery_prepare_tensor_v2() {
  std::lock_guard<std::mutex> guard(launch_mutex);
  if(cleanup_error) return cleanup_error;
  aclrtContext current=nullptr;
  auto status=aclrtGetCurrentContext(&current);
  if(status) return status;
  if(!current) return -10;
  if(ready) return current==bound_context ? 0 : -12;
  for(auto& v:variants) {
    aclrtBinaryLoadOption options[]={
      {.type=ACL_RT_BINARY_LOAD_OPT_LAZY_LOAD,.value={.isLazyLoad=0}},
      {.type=ACL_RT_BINARY_LOAD_OPT_MAGIC,.value={.magic=v.magic}}
    };
    aclrtBinaryLoadOptions opts={.options=options,.numOpt=2};
    status=aclrtBinaryLoadFromData(v.image,v.size,&opts,&v.binary);
    if(!status && !v.binary) status=-14;
    if(!status) status=aclrtBinaryGetFunction(v.binary,v.symbol,&v.function);
    if(!status && !v.function) status=-15;
    if(status) {cleanup_error=release_partial();return cleanup_error ? cleanup_error : status;}
  }
  bound_context=current; ready=true;
  return 0;
}
"""
