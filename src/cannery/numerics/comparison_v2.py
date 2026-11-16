"""Mixed-dtype comparison outside the candidate process; integers stay exact."""

import math
import struct
from typing import Any

from cannery.numerics.comparison import compare
from cannery.numerics.recipes import MAX_TENSOR_ELEMENTS
from cannery.numerics.recipes_v2 import ITEM_BYTES, DType


def compare_tensor(
    reference: bytes,
    candidate: bytes,
    dtype: DType,
    *,
    rtol: float,
    atol: float,
    equal_nan: bool = False,
) -> dict[str, Any]:
    """Compare logical raw bytes; callers separately verify shape and dtype.

    Floating values use v1 torch-close semantics. Integer and boolean outputs
    require exact values regardless of tolerances; int64 is never cast to float.
    """
    if (
        dtype not in ITEM_BYTES
        or type(reference) is not bytes
        or type(candidate) is not bytes
        or len(reference) != len(candidate)
        or len(reference) % ITEM_BYTES[dtype]
        or len(reference) > MAX_TENSOR_ELEMENTS * ITEM_BYTES[dtype]
    ):
        raise ValueError("Invalid mixed-dtype comparison payload")
    if type(equal_nan) is not bool or any(
        type(v) not in (float, int) or not math.isfinite(v) or v < 0 for v in (rtol, atol)
    ):
        raise ValueError("Invalid numerical tolerance")
    if dtype in {"float32", "float16", "bfloat16"}:

        def f32(data: bytes) -> bytes:
            if dtype == "float32":
                return data
            if dtype == "bfloat16":
                return b"".join(b"\0\0" + data[i : i + 2] for i in range(0, len(data), 2))
            return b"".join(struct.pack("<f", value) for (value,) in struct.iter_unpack("<e", data))

        return {
            "metric": "torch_close",
            **compare(
                f32(reference),
                f32(candidate),
                rtol=rtol,
                atol=atol,
                equal_nan=equal_nan,
            ).as_dict(),
        }
    if dtype == "bool" and (any(v > 1 for v in reference) or any(v > 1 for v in candidate)):
        raise ValueError("Boolean comparison requires canonical storage bytes")
    code = {"int8": "b", "int32": "i", "int64": "q", "bool": "B"}[dtype]
    mismatches = maximum = 0
    for (expected,), (actual,) in zip(
        struct.iter_unpack("<" + code, reference),
        struct.iter_unpack("<" + code, candidate),
        strict=True,
    ):
        mismatches += expected != actual
        maximum = max(maximum, abs(expected - actual))
    return {
        "metric": "exact",
        "passed": mismatches == 0,
        "elements": len(reference) // ITEM_BYTES[dtype],
        "mismatches": mismatches,
        "max_absolute_error": maximum,
    }
