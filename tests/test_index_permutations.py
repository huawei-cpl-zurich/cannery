import inspect
import json
import struct
import subprocess
import sys

import pytest

from cannery.numerics.recipes_v2 import (
    ITEM_BYTES,
    TensorRecipeV2,
    _generate_v2,
    generate_v2,
    input_sha256_v2,
)


def permutation(count, dtype="int32", seed=42):
    return TensorRecipeV2(shape=(count,), dtype=dtype, distribution="permutation", seed=seed)


@pytest.mark.parametrize("count", [0, 1, 2, 16, 33, 1024])
@pytest.mark.parametrize("dtype,code", [("int32", "i"), ("int64", "q")])
def test_reproducible_unique_indices_cover_exactly_the_declared_domain(count, dtype, code):
    recipe = permutation(count, dtype)
    raw = generate_v2(recipe, "small", "indices")
    values = struct.unpack(f"<{count}{code}", raw)
    assert sorted(values) == list(range(count))
    assert raw == generate_v2(recipe, "small", "indices")
    assert len(raw) == recipe.nbytes
    assert input_sha256_v2(recipe, raw) == input_sha256_v2(recipe, bytes(raw))


def test_portable_golden_vector_matches_both_integer_storage_formats():
    expected = (10, 6, 2, 14, 15, 13, 12, 8, 1, 3, 11, 5, 4, 9, 7, 0)
    for dtype, code in [("int32", "i"), ("int64", "q")]:
        assert generate_v2(permutation(16, dtype), "small", "indices") == struct.pack(
            "<16" + code, *expected
        )


def test_framing_and_seed_separate_deterministic_streams():
    recipe = permutation(32)
    before = generate_v2(recipe, "small", "indices")
    assert before != generate_v2(recipe, "other", "indices")
    assert before != generate_v2(recipe, "small", "other")
    assert before != generate_v2(permutation(32, seed=43), "small", "indices")


@pytest.mark.parametrize(
    "change",
    [
        {"dtype": "float32"},
        {"dtype": "int8"},
        {"dtype": "bool"},
        {"shape": (2, 3)},
        {"layout": "transpose"},
        {"low": 0},
        {"high": 3},
        {"shape": (1048577,)},
    ],
)
def test_unsupported_permutations_are_rejected_before_generation(change):
    with pytest.raises(ValueError):
        TensorRecipeV2.model_validate(permutation(8).model_dump() | change)


def test_trusted_supervisor_snippet_requires_only_stdlib_and_matches_controller():
    recipe = permutation(33, "int64")
    # Oracle, numerical and benchmark supervisors embed this same generator.
    program = (
        "from __future__ import annotations\nimport hashlib,math,struct,json\n"
        "from types import SimpleNamespace\n"
        f"ITEM_BYTES={ITEM_BYTES!r}\n"
        + inspect.getsource(_generate_v2)
        + f"values=json.loads({json.dumps(recipe.model_dump(mode='json'))!r})\n"
        + 'print(_generate_v2(SimpleNamespace(**values),"small","indices").hex())\n'
    )
    result = subprocess.check_output([sys.executable, "-I", "-c", program], text=True, timeout=5)
    assert result.strip() == generate_v2(recipe, "small", "indices").hex()


@pytest.mark.parametrize("dtype,code", [("int32", "i"), ("int64", "q")])
def test_inclusive_integer_sequence_generates_reproducible_group_prefixes(dtype, code):
    recipe = TensorRecipeV2(
        shape=(5,), dtype=dtype, distribution="sequence", seed=0, low=0, high=256
    )
    assert generate_v2(recipe, "grouped", "group_list") == struct.pack(
        "<5" + code, 0, 64, 128, 192, 256
    )
    negative = recipe.model_copy(update={"low": -128, "high": 128})
    assert generate_v2(negative, "grouped", "group_list") == struct.pack(
        "<5" + code, -128, -64, 0, 64, 128
    )


@pytest.mark.parametrize("count", [0, 1])
def test_empty_and_singleton_sequences_are_bounded(count):
    recipe = TensorRecipeV2(
        shape=(count,), dtype="int64", distribution="sequence", seed=0, low=7, high=7
    )
    assert generate_v2(recipe, "case", "indices") == struct.pack("<q", 7) * count


@pytest.mark.parametrize(
    "change",
    [
        {"dtype": "float32"},
        {"dtype": "int8"},
        {"shape": (2, 2)},
        {"low": None},
        {"high": None},
        {"low": 5, "high": 0},
        {"high": 7},
        {"shape": (1,)},
    ],
)
def test_fractional_invalid_or_unbounded_sequences_are_rejected(change):
    with pytest.raises(ValueError):
        TensorRecipeV2.model_validate(
            dict(shape=(5,), dtype="int64", distribution="sequence", seed=0, low=0, high=256)
            | change
        )
