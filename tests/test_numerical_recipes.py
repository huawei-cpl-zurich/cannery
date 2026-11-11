import hashlib
import struct
import subprocess
import sys

import pytest

from cannery.numerics.recipes import (
    RecipeSuite,
    TensorRecipe,
    generate,
    input_sha256,
    parse_recipes,
)


def recipe(**updates):
    return TensorRecipe.model_validate(
        {"shape": [17], "distribution": "uniform_f32", "seed": 42, **updates}
    )


def test_portable_uniform_golden_vector_and_process_replay():
    item = recipe(shape=[8])
    data = generate(item, "small", "x")
    assert data.hex() == "72ff08bfb8010abf0679133fd0aaac3e80942f3ed898dcbe902bab3eb2de13bf"
    assert all(-1 <= v < 1 for v in struct.unpack("<8f", data))
    script = (
        "from cannery.numerics.recipes import *;"
        f"r=TensorRecipe.model_validate_json({item.model_dump_json()!r});"
        "print(generate(r,'small','x').hex())"
    )
    for _ in range(2):
        assert (
            subprocess.check_output([sys.executable, "-c", script], text=True).strip() == data.hex()
        )


def test_seed_case_name_and_input_name_have_independent_streams():
    baseline = generate(recipe(), "small", "x")
    assert baseline != generate(recipe(seed=43), "small", "x")
    assert baseline != generate(recipe(), "different", "x")
    assert baseline != generate(recipe(), "small", "other")
    assert baseline == generate(recipe(), "small", "x")


def test_arange_transpose_and_empty_have_exact_logical_values():
    item = recipe(shape=[4, 3], distribution="arange_f32", layout="transpose")
    data = generate(item, "layout", "x")
    assert struct.unpack("<12f", data) == (0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11)
    assert generate(recipe(shape=[0, 3]), "empty", "x") == b""
    assert generate(recipe(shape=[3], distribution="zeros_f32"), "zero", "x") == bytes(12)
    assert len(generate(recipe(), "tail", "x")) == 68


@pytest.mark.parametrize(
    "change",
    [
        {"shape": []},
        {"shape": [1] * 5},
        {"shape": [-1]},
        {"shape": [True]},
        {"shape": [1.5]},
        {"shape": [1048577]},
        {"shape": [1024, 1025]},
        {"dtype": "float16"},
        {"distribution": "torch_randn"},
        {"seed": -1},
        {"seed": 2**64},
        {"seed": True},
        {"layout": "transpose"},
        {"path": "/secret"},
    ],
)
def test_invalid_or_unbounded_recipes_fail_before_generation(change):
    with pytest.raises(ValueError):
        recipe(**change)


def suite(cases):
    return {"schema_version": 1, "generator": "sha256-f32-v1", "cases": cases}


def test_suite_identity_and_budget_are_frozen_and_bounded():
    case = {"id": "one", "inputs": {"x": recipe().model_dump(mode="json")}}
    value = RecipeSuite.model_validate(suite([case]))
    assert value.sha256 == hashlib.sha256(value.canonical_bytes()).hexdigest()
    assert parse_recipes(value.canonical_bytes()) == value
    for cases in [[], [case] * 2, [dict(case, id=f"case_{i}") for i in range(33)]]:
        with pytest.raises(ValueError):
            RecipeSuite.model_validate(suite(cases))
    large = recipe(shape=[1048576]).model_dump(mode="json")
    with pytest.raises(ValueError, match="aggregate"):
        RecipeSuite.model_validate(
            suite([{"id": f"case_{i}", "inputs": {"x": large}} for i in range(9)])
        )
    with pytest.raises(ValueError):
        parse_recipes(b"schema_version: 1\nschema_version: 1\n")


def test_input_digest_binds_shape_layout_and_values():
    item = recipe(shape=[2, 3], distribution="zeros_f32")
    data = generate(item, "zeros", "x")
    original = input_sha256(item, data)
    for other in [
        item.model_copy(update={"shape": (3, 2)}),
        item.model_copy(update={"layout": "transpose"}),
    ]:
        assert input_sha256(other, data) != original
    assert input_sha256(item, bytes([1]) + data[1:]) != original
    with pytest.raises(ValueError):
        input_sha256(item, data[:-1])
    with pytest.raises(ValueError):
        generate(item, "bad\0name", "x")
    with pytest.raises(ValueError):
        generate(item, "valid", "x\0bad")
