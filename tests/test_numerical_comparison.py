import json
import math
import struct

import pytest

from cannery.numerics.comparison import compare


def raw(*values):
    return struct.pack("<" + "f" * len(values), *values)


def test_compound_predicate_uses_each_reference_value():
    ref = raw(1, 100)
    got = raw(1.5, 100.5)
    assert not compare(ref, got, rtol=0.01, atol=0).passed
    result = compare(ref, got, rtol=0.01, atol=0.5)
    assert result.passed and result.max_absolute_error == 0.5
    assert result.max_relative_error == 0.5 and result.rms_error == 0.5
    assert result.elements == 2 and result.finite_pairs == 2


def test_zero_reference_policy_is_explicit_and_json_safe():
    result = compare(raw(0, 0), raw(0, 0.25), rtol=100, atol=0)
    assert not result.passed and result.mismatches == 1
    assert result.zero_reference_mismatches == 1 and result.max_relative_error is None
    assert compare(raw(-0.0), raw(0), rtol=0, atol=0).max_relative_error == 0
    json.dumps(result.as_dict(), allow_nan=False)


@pytest.mark.parametrize(
    "expected,actual,equal_nan,passed",
    [
        (float("nan"), float("nan"), False, False),
        (float("nan"), float("nan"), True, True),
        (float("nan"), 1, True, False),
        (float("inf"), float("inf"), False, True),
        (float("-inf"), float("inf"), False, False),
        (1, float("inf"), False, False),
        (float("inf"), 1, False, False),
    ],
)
def test_nonfinite_matching_and_counts(expected, actual, equal_nan, passed):
    result = compare(raw(expected), raw(actual), rtol=1, atol=1, equal_nan=equal_nan)
    assert result.passed == passed
    assert result.reference_nonfinite == int(not math.isfinite(expected))
    assert result.candidate_nonfinite == int(not math.isfinite(actual))
    assert result.nonfinite_mismatches == int(not passed)
    assert result.finite_pairs == 0
    json.dumps(result.as_dict(), allow_nan=False)


def test_empty_inputs_and_full_float32_range_have_finite_metrics():
    result = compare(b"", b"", rtol=0, atol=0)
    assert result.passed and result.elements == 0 and result.rms_error == 0
    result = compare(raw(3e38, 1e-40), raw(-3e38, 1), rtol=0, atol=0)
    assert result.mismatches == 2 and math.isfinite(result.rms_error)
    json.dumps(result.as_dict(), allow_nan=False)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rtol": -1},
        {"atol": float("nan")},
        {"rtol": float("inf")},
        {"rtol": True},
        {"equal_nan": 1},
    ],
)
def test_invalid_tolerances_are_never_interpreted_as_success(kwargs):
    with pytest.raises(ValueError):
        compare(raw(1), raw(1), **({"rtol": 0, "atol": 0} | kwargs))


@pytest.mark.parametrize(
    "left,right",
    [
        (b"a", b"a"),
        (raw(1), b""),
        (bytearray(4), bytes(4)),
        (bytes(4 * 1048577), bytes(4 * 1048577)),
    ],
)
def test_corrupt_truncated_or_oversized_outputs_rejected(left, right):
    with pytest.raises(ValueError):
        compare(left, right, rtol=0, atol=0)


def test_tolerance_results_are_specific_not_inferred_from_summary_errors():
    left, right = raw(1, 100), raw(1.25, 100.25)
    strict = compare(left, right, rtol=0, atol=0.1)
    loose = compare(left, right, rtol=0, atol=0.3)
    assert not strict.passed and loose.passed
    assert strict.max_absolute_error == loose.max_absolute_error
    assert strict.atol != loose.atol
