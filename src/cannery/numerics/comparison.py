"""Bounded float32 comparison outside the untrusted candidate process."""

import math
import struct
from dataclasses import asdict, dataclass
from typing import Any

from cannery.numerics.recipes import MAX_TENSOR_ELEMENTS


@dataclass(frozen=True)
class Comparison:
    passed: bool
    elements: int
    finite_pairs: int
    mismatches: int
    reference_nonfinite: int
    candidate_nonfinite: int
    nonfinite_mismatches: int
    zero_reference_mismatches: int
    max_absolute_error: float
    max_relative_error: float | None
    rms_error: float
    rtol: float
    atol: float
    equal_nan: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def compare(
    reference: bytes, candidate: bytes, *, rtol: float, atol: float, equal_nan: bool = False
) -> Comparison:
    """Compare raw logical float32 values; caller verifies shape/dtype separately.

    Metrics describe finite pairs only. Relative error is zero for identical
    zeros and null if a zero reference has a nonzero candidate. Nonfinite pairs
    have explicit counters, never fabricated numeric errors or JSON infinities.
    """
    if (
        type(reference) is not bytes
        or type(candidate) is not bytes
        or len(reference) != len(candidate)
        or len(reference) % 4
        or len(reference) > MAX_TENSOR_ELEMENTS * 4
    ):
        raise ValueError("Invalid float32 comparison payload")
    if (
        any(
            type(value) not in (float, int) or not math.isfinite(value) or value < 0
            for value in (rtol, atol)
        )
        or type(equal_nan) is not bool
    ):
        raise ValueError("Invalid numerical tolerance")
    finite = mismatches = ref_nonfinite = got_nonfinite = nonfinite_mismatches = zero = 0
    maximum = relative = squared = 0.0
    for (expected,), (actual,) in zip(
        struct.iter_unpack("<f", reference), struct.iter_unpack("<f", candidate), strict=True
    ):
        ref_finite, got_finite = math.isfinite(expected), math.isfinite(actual)
        ref_nonfinite += not ref_finite
        got_nonfinite += not got_finite
        if not ref_finite or not got_finite:
            matched = expected == actual or (
                equal_nan and math.isnan(expected) and math.isnan(actual)
            )
            if not matched:
                nonfinite_mismatches += 1
                mismatches += 1
            continue
        finite += 1
        error = abs(actual - expected)
        mismatches += error > atol + rtol * abs(expected)
        maximum = max(maximum, error)
        squared += error * error
        if expected == 0:
            zero += error != 0
        else:
            relative = max(relative, error / abs(expected))
    return Comparison(
        passed=mismatches == 0,
        elements=len(reference) // 4,
        finite_pairs=finite,
        mismatches=mismatches,
        reference_nonfinite=ref_nonfinite,
        candidate_nonfinite=got_nonfinite,
        nonfinite_mismatches=nonfinite_mismatches,
        zero_reference_mismatches=zero,
        max_absolute_error=maximum,
        max_relative_error=None if zero else relative,
        rms_error=math.sqrt(squared / finite) if finite else 0.0,
        rtol=float(rtol),
        atol=float(atol),
        equal_nan=equal_nan,
    )
