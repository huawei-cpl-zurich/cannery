"""Verified pointer-only Ascend950 SIMD Triton exports; no runtime compiler dependency."""

import math
import re
from typing import Any

from cannery.builds.ascendc import BuildError
from cannery.builds.catlass import _native_object
from cannery.builds.catlass_tensor import StaticTensorExport


def validate_triton_export(
    binary: bytes,
    metadata: dict[str, Any],
    signature: dict[str, Any],
    bindings: list[list[str]],
    grid: tuple[int, int, int],
    *,
    arch: str,
) -> StaticTensorExport:
    """Pinned backend launch layout: no FFTS, workspace, sync locks, SIMT or grid transforms.

    The producer compiles a pointer-only ASTSource derived from these bindings;
    all source scalar parameters specialize at build time. Mixed Cube/vector
    binaries use AICORE load magic, matching this pinned Triton backend's loader.
    """
    try:
        if (
            not _native_object(binary)
            or not arch.startswith("Ascend950")
            or metadata["arch"] != arch
            or metadata["target_arch"] != arch
            or metadata["compile_on_910_95"] is not True
            or metadata["is_pure_simt"] is not False
            or metadata["parallel_mode"] != "simd"
            or metadata["debug"] is not False
            or metadata["mix_mode"] not in ("mix", "aic", "aiv")
            or metadata["name"] != metadata["kernel_name"]
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", metadata["name"]) is None
            or metadata["program_grid_mapping_applied"] is not False
            or metadata["row_coalescing_applied"] is not False
            or metadata["program_grid_transforms"] is not None
            or metadata["cluster_dims"] != [1, 1, 1]
            or any(type(v) is not int for v in metadata["cluster_dims"])
            or type(metadata["num_ctas"]) is not int
            or metadata["num_ctas"] != 1
            or metadata["launch_cooperative_grid"] is not False
            or metadata["instrumentation_mode"] != ""
            or type(grid) is not tuple
            or len(grid) != 3
            or any(type(v) is not int or not 1 <= v <= 65535 for v in grid)
            or math.prod(grid) > 65535
            or type(bindings) is not list
            or not 1 <= len(bindings) <= 64
        ):
            raise ValueError
        for field, default, allowed in [
            ("workspace_size", -1, (-1, 0)),
            ("sync_block_lock_layout", 0, (0,)),
        ]:
            value = metadata.get(field, default)
            if type(value) is not int or value not in allowed:
                raise ValueError
        kinds = metadata["tensor_kinds"]
        if (
            type(kinds) is not list
            or len(kinds) != len(bindings)
            or any(type(k) is not int or k not in (0, 1, 2) for k in kinds)
        ):
            raise ValueError
        resolved = []
        for binding in bindings:
            if (
                type(binding) is not list
                or len(binding) != 2
                or binding[0] not in ("input", "output")
                or type(binding[1]) is not str
            ):
                raise ValueError
            direction, name = binding
            index = next(i for i, t in enumerate(signature[direction + "s"]) if t["name"] == name)
            resolved.append((direction, index))
        for index, output in enumerate(signature["outputs"]):
            if ("output", index) in resolved:
                continue
            alias = output["alias"]
            original = next(i for i, t in enumerate(signature["inputs"]) if t["name"] == alias)
            if ("input", original) not in resolved:
                raise ValueError
        return StaticTensorExport(
            binary,
            metadata["name"],
            "aiv" if metadata["mix_mode"] == "aiv" else "aic",
            math.prod(grid),
            tuple(resolved),
        )
    except (ValueError, TypeError, KeyError, StopIteration, AttributeError):
        raise BuildError("Unsupported static Triton export or launch ABI") from None
