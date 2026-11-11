"""Validate static CATLASS exports and generate a source-free native ACL bridge."""

import re
from dataclasses import dataclass
from typing import Any

from cannery.builds.ascendc import BuildError

MAX_OBJECT_BYTES = 8 * 1024 * 1024


def _native_object(binary: bytes) -> bool:
    # Observed ELF contract of the pinned A5 compiler, not a host shared library.
    return (
        type(binary) is bytes
        and 64 <= len(binary) <= MAX_OBJECT_BYTES
        and binary[:7] == b"\x7fELF\x02\x01\x01"
        and int.from_bytes(binary[16:18], "little") == 2
        and int.from_bytes(binary[18:20], "little") == 4137
    )


@dataclass(frozen=True)
class NativeVariant:
    count: int
    entrypoint: str
    binary: bytes


def validate_export(count: int, binary: bytes, metadata: dict[str, Any]) -> NativeVariant:
    """Admit only an AIV kernel whose public launch ABI is two static pointers."""
    if type(count) is not int or not 1 <= count <= 1048576 or not _native_object(binary):
        raise BuildError("Invalid CATLASS native object")
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
        entrypoint = metadata["entrypoint"]
        if (
            not isinstance(entrypoint, str)
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", entrypoint) is None
            or metadata["kernel_mode"] != "aiv"
            or any(
                metadata[key] is not False
                for key in ("uses_scalar_print", "uses_tensor_print", "is_mixed")
            )
            or type(metadata["ub_static_bytes"]) is not int
            or not 0 <= metadata["ub_static_bytes"] <= 262144
            or type(metadata["ub_dynamic_base"]) is not int
            or metadata["ub_dynamic_base"] != -1
        ):
            raise ValueError
        abi = metadata["kernel_abi"]
        if (
            set(abi) != {"schema_version", "entrypoint", "total_size", "arguments"}
            or type(abi["schema_version"]) is not int
            or abi["schema_version"] != 3
            or abi["entrypoint"] != entrypoint
            or type(abi["total_size"]) is not int
            or abi["total_size"] != 16
            or not isinstance(abi["arguments"], list)
            or len(abi["arguments"]) != 2
        ):
            raise ValueError
        for index, argument in enumerate(abi["arguments"]):
            expected = {
                "index": index,
                "logical_index": index,
                "kind": "pointer",
                "field": None,
                "scalar": None,
                "offset": 8 * index,
                "storage_size": 8,
                "alignment": 4,
            }
            if set(argument) != set(expected) | {"mlir_type"}:
                raise ValueError
            for key, value in expected.items():
                if argument[key] != value or type(argument[key]) is not type(value):
                    raise ValueError
            if argument["mlir_type"] != f"memref<{count}xf32, #hivm.address_space<gm>>":
                raise ValueError
        return NativeVariant(count, entrypoint, binary)
    except (KeyError, TypeError, ValueError):
        raise BuildError("Unsupported CATLASS export ABI") from None


def native_wrapper(variants: list[NativeVariant]) -> str:
    """Embed verified device objects; runtime needs AscendCL, never CATLASS/JIT."""
    import json

    if not 1 <= len(variants) <= 4 or len({v.count for v in variants}) != len(variants):
        raise BuildError("Expected one to four distinct nonempty CATLASS variants")
    declarations, records = [], []
    for index, variant in enumerate(sorted(variants, key=lambda item: item.count)):
        if (
            type(variant.count) is not int
            or not 1 <= variant.count <= 1048576
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", variant.entrypoint) is None
            or not _native_object(variant.binary)
        ):
            raise BuildError("Invalid native variant")
        declarations.append(
            f"alignas(64) static const unsigned char object_{index}[] = {{"
            + ",".join(str(value) for value in variant.binary)
            + "};"
        )
        records.append(
            f"{{{variant.count}, object_{index}, sizeof(object_{index}), "
            + json.dumps(variant.entrypoint)
            + "}"
        )
    return (
        "#include <acl/acl.h>\n#include <cstdint>\n#include <mutex>\n"
        + "\n".join(declarations)
        + "\nstruct Variant { uint64_t count; const unsigned char* data; size_t size;\n"
        + "const char* symbol; aclrtBinHandle binary{}; aclrtFuncHandle function{};\n"
        + "bool loaded=false; int error=0; };\nstatic Variant variants[] = {"
        + ",".join(records)
        + "};\n"
        + ACL_BRIDGE
    )


ACL_BRIDGE = r"""
static std::mutex launch_mutex;
static aclrtContext bound_context = nullptr;
extern "C" int cannery_launch(void* stream, void* input, void* output, uint64_t count) {
    if (count == 0) return 0;
    std::lock_guard<std::mutex> guard(launch_mutex);
    Variant* selected = nullptr;
    for (auto& variant : variants) if (variant.count == count) selected = &variant;
    if (!selected) return -1;
    aclrtContext context = nullptr;
    const auto context_status = aclrtGetCurrentContext(&context);
    if (context_status) return static_cast<int>(context_status);
    if (!context || (bound_context && bound_context != context)) return -2;
    bound_context = context;
    auto& v = *selected;
    if (v.error) return v.error;
    auto check = [&v](aclError status) { v.error = static_cast<int>(status); return v.error; };
    if (!v.loaded) {
        aclrtBinaryLoadOption option{};
        option.type = ACL_RT_BINARY_LOAD_OPT_MAGIC;
        option.value.magic = ACL_RT_BINARY_MAGIC_ELF_VECTOR_CORE;
        aclrtBinaryLoadOptions options{&option, 1};
        if (check(aclrtBinaryLoadFromData(v.data, v.size, &options, &v.binary))) return v.error;
        if (check(aclrtBinaryGetFunction(v.binary, v.symbol, &v.function))) return v.error;
        v.loaded = true;
    }
    aclrtArgsHandle args{};
    aclrtParamHandle parameter{};
    void* pointers[] = {input, output};
    if (check(aclrtKernelArgsInit(v.function, &args))) return v.error;
    if (check(aclrtKernelArgsAppend(args, pointers, sizeof(pointers), &parameter))) return v.error;
    if (check(aclrtKernelArgsFinalize(args))) return v.error;
    aclrtLaunchKernelCfg config{};
    return check(aclrtLaunchKernelWithConfig(v.function, 1,
        reinterpret_cast<aclrtStream>(stream), &config, args, nullptr));
}
"""
