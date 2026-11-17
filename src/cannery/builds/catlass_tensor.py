"""Validate static multi-tensor CATLASS objects without importing the compiler."""

import re
from dataclasses import dataclass
from typing import Any, Literal

from cannery.builds.ascendc import BuildError
from cannery.builds.catlass import _native_object

MLIR_TYPES = {
    "float32": "f32",
    "float16": "f16",
    "bfloat16": "bf16",
    "int8": "i8",
    "int32": "i32",
    "int64": "i64",
    "bool": "i1",
}


@dataclass(frozen=True)
class StaticTensorExport:
    binary: bytes
    entrypoint: str
    mode: Literal["aic", "aiv"]
    block_num: int
    # Each native pointer addresses one ordered input or output descriptor.
    bindings: tuple[tuple[str, int], ...]


def validate_static_export(
    binary: bytes,
    metadata: dict[str, Any],
    signature: dict[str, Any],
    bindings: list[list[str]],
    block_num: int,
) -> StaticTensorExport:
    """Resolve pointer bindings against a trusted frozen tensor signature.

    Only static pointer ABIs are admitted. Dynamic memref fields, runtime scalar
    parameters, mixed kernels, debug printing and dynamic UB require separate
    verified support. Scalar values in the frozen signature specialize the build.
    """
    if not _native_object(binary) or type(block_num) is not int or not 1 <= block_num <= 65535:
        raise BuildError("Invalid static tensor object or block count")
    try:
        if set(metadata) != {
            "entrypoint",
            "kernel_mode",
            "kernel_abi",
            "uses_scalar_print",
            "uses_tensor_print",
            "is_mixed",
            "ub_static_bytes",
            "ub_dynamic_base",
        }:
            raise ValueError
        symbol, mode = metadata["entrypoint"], metadata["kernel_mode"]
        if (
            not isinstance(symbol, str)
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", symbol) is None
            or mode not in ("aic", "aiv")
            or any(
                metadata[name] is not False
                for name in ("uses_scalar_print", "uses_tensor_print", "is_mixed")
            )
            or type(metadata["ub_static_bytes"]) is not int
            or not 0 <= metadata["ub_static_bytes"] <= 253952
            or type(metadata["ub_dynamic_base"]) is not int
            or metadata["ub_dynamic_base"] != -1
            or type(bindings) is not list
            or not 1 <= len(bindings) <= 64
        ):
            raise ValueError
        abi = metadata["kernel_abi"]
        if (
            set(abi) != {"schema_version", "entrypoint", "total_size", "arguments"}
            or type(abi["schema_version"]) is not int
            or abi["schema_version"] != 3
            or abi["entrypoint"] != symbol
            or type(abi["total_size"]) is not int
            or abi["total_size"] != len(bindings) * 8
            or type(abi["arguments"]) is not list
            or len(abi["arguments"]) != len(bindings)
        ):
            raise ValueError
        resolved = []
        for index, (binding, argument) in enumerate(zip(bindings, abi["arguments"], strict=True)):
            if (
                type(binding) is not list
                or len(binding) != 2
                or binding[0] not in ("input", "output")
                or type(binding[1]) is not str
            ):
                raise ValueError
            direction, name = binding
            tensors = signature[direction + "s"]
            position = next(i for i, tensor in enumerate(tensors) if tensor["name"] == name)
            descriptor = tensors[position]
            shape = "x".join(str(d) for d in descriptor["shape"])
            expected = {
                "index": index,
                "kind": "pointer",
                "field": None,
                "scalar": None,
                "offset": index * 8,
                "storage_size": 8,
                "alignment": 4,
                "mlir_type": f"memref<{shape}x{MLIR_TYPES[descriptor['dtype']]}, "
                "#hivm.address_space<gm>>",
            }
            if (
                set(argument) != set(expected) | {"logical_index"}
                or any(
                    type(argument[key]) is not type(value) or argument[key] != value
                    for key, value in expected.items()
                )
                or type(argument["logical_index"]) is not int
                or not 0 <= argument["logical_index"] < 128
            ):
                raise ValueError
            resolved.append((direction, position))
        for index, output in enumerate(signature["outputs"]):
            if ("output", index) in resolved:
                continue
            alias = output["alias"]
            if alias is None or not any(
                direction == "input" and signature["inputs"][position]["name"] == alias
                for direction, position in resolved
            ):
                raise ValueError
        return StaticTensorExport(binary, symbol, mode, block_num, tuple(resolved))
    except (ValueError, TypeError, KeyError, StopIteration):
        raise BuildError("Unsupported static tensor export ABI or bindings") from None
