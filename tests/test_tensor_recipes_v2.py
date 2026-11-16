import struct
import subprocess
import sys

import pytest

from cannery.numerics.recipes import parse_recipes
from cannery.numerics.recipes_v2 import (
    ITEM_BYTES,
    RecipeSuiteV2,
    TensorRecipeV2,
    _float_bytes,
    generate_v2,
    input_sha256_v2,
    parse_recipes_v2,
)


def recipe(**changes):
    return TensorRecipeV2.model_validate(
        dict(shape=[8], dtype="float32", distribution="uniform", seed=42) | changes
    )


def test_golden_vector_and_process_replay():
    r = recipe()
    data = generate_v2(r, "small", "x")
    assert data.hex() == "78991c3f307e573f90b518be40c9d3bc561333bf14c4a03e60546c3e82af7c3f"
    script = (
        "from cannery.numerics.recipes_v2 import *;"
        f"r=TensorRecipeV2.model_validate_json({r.model_dump_json()!r});"
        "print(generate_v2(r,'small','x').hex())"
    )
    assert subprocess.check_output([sys.executable, "-c", script], text=True).strip() == data.hex()
    assert data != generate_v2(recipe(seed=43), "small", "x")
    assert data != generate_v2(r, "other", "x")
    assert data != generate_v2(r, "small", "y")


@pytest.mark.parametrize("dtype,code", [("float16", "e"), ("float32", "f")])
def test_floating_storage_matches_ieee_rounding(dtype, code):
    f32 = struct.unpack("<8f", generate_v2(recipe(), "small", "x"))
    assert generate_v2(recipe(dtype=dtype), "small", "x") == struct.pack("<8" + code, *f32)


def test_bfloat_rounding_even_and_odd_ties():
    assert _float_bytes(1 + 2**-8, "bfloat16").hex() == "803f"
    assert _float_bytes(1 + 3 * 2**-8, "bfloat16").hex() == "823f"
    assert _float_bytes(-1 - 3 * 2**-8, "bfloat16").hex() == "82bf"
    assert len(generate_v2(recipe(dtype="bfloat16"), "small", "x")) == 16


@pytest.mark.parametrize(
    "dtype,low,high",
    [
        ("int8", -128, 127),
        ("int32", -(2**31), 2**31 - 1),
        ("int64", 2**63 - 8, 2**63 - 1),
        ("bool", 0, 1),
    ],
)
def test_integer_bounds_and_storage(dtype, low, high):
    r = recipe(dtype=dtype, distribution="randint", low=low, high=high, shape=[17])
    data = generate_v2(r, "small", "indices")
    width = ITEM_BYTES[dtype]
    assert len(data) == 17 * width
    values = [
        int.from_bytes(data[i : i + width], "little", signed=dtype != "bool")
        for i in range(0, len(data), width)
    ]
    assert all(low <= value <= high for value in values)
    fixed = recipe(dtype=dtype, distribution="randint", low=low, high=low)
    assert (
        generate_v2(fixed, "small", "x")
        == low.to_bytes(width, "little", signed=dtype != "bool") * 8
    )


@pytest.mark.parametrize("dtype", list(ITEM_BYTES))
def test_zeros_empty_and_transpose(dtype):
    assert generate_v2(recipe(dtype=dtype, distribution="zeros"), "c", "x") == bytes(
        8 * ITEM_BYTES[dtype]
    )
    assert generate_v2(recipe(dtype=dtype, distribution="zeros", shape=[0]), "c", "x") == b""
    if dtype.startswith("int"):
        r = recipe(dtype=dtype, shape=[2, 3], distribution="randint", low=0, high=100)
    else:
        r = recipe(
            dtype=dtype, shape=[2, 3], distribution="zeros" if dtype == "bool" else "uniform"
        )
    data = generate_v2(r, "c", "x")
    width = ITEM_BYTES[dtype]
    expected = b"".join(data[i * width : (i + 1) * width] for i in [0, 2, 4, 1, 3, 5])
    assert generate_v2(r.model_copy(update={"layout": "transpose"}), "c", "x") == expected


@pytest.mark.parametrize(
    "change",
    [
        {"shape": [True]},
        {"shape": [1024, 1025]},
        {"shape": []},
        {"shape": [1] * 5},
        {"dtype": "float64"},
        {"seed": True},
        {"distribution": "randint"},
        {"low": 0},
        {"layout": "transpose"},
        {"dtype": "int8", "distribution": "randint", "low": -129, "high": 0},
        {"dtype": "int64", "distribution": "randint", "low": 0, "high": 2**32},
        {"dtype": "bool", "distribution": "uniform"},
        {"dtype": "int8", "distribution": "randint", "low": 1, "high": 0},
    ],
)
def test_invalid_recipes_rejected_before_generation(change):
    with pytest.raises(ValueError):
        recipe(**change)


def test_suite_roundtrip_budget_identity_and_v1_gate():
    case = {
        "id": "small",
        "inputs": {
            "x": recipe().model_dump(mode="json"),
            "weight": recipe(dtype="float16").model_dump(mode="json"),
        },
    }
    suite = RecipeSuiteV2.model_validate(
        dict(schema_version=2, generator="sha256-tensor-v2", cases=[case])
    )
    assert parse_recipes_v2(suite.canonical_bytes()) == suite
    with pytest.raises(ValueError):
        parse_recipes(suite.canonical_bytes())
    with pytest.raises(ValueError, match="Duplicate"):
        RecipeSuiteV2.model_validate(suite.model_dump() | {"cases": (suite.cases[0],) * 2})
    big = recipe(dtype="int64", distribution="zeros", shape=[1048576])
    case["inputs"] = {f"x{i}": big.model_dump(mode="json") for i in range(5)}
    with pytest.raises(ValueError, match="aggregate"):
        RecipeSuiteV2.model_validate(
            dict(schema_version=2, generator="sha256-tensor-v2", cases=[case])
        )
    r = recipe()
    data = generate_v2(r, "c", "x")
    assert input_sha256_v2(r, data) != input_sha256_v2(recipe(shape=[2, 4]), data)
    with pytest.raises(ValueError, match="length"):
        input_sha256_v2(r, b"")
    with pytest.raises(ValueError):
        generate_v2(r, "c\0other", "x")
