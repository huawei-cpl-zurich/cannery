import copy
import struct

import pytest

from cannery.numerics.comparison_v2 import compare_tensor
from cannery.numerics.recipes_v2 import RecipeSuiteV2
from cannery.numerics.tensor_contract import ScalarValueV2, TensorContractV2, parse_tensor_contract


def fixture():
    suite = {
        "schema_version": 2,
        "generator": "sha256-tensor-v2",
        "cases": [
            {
                "id": "small",
                "inputs": {
                    "x": {
                        "shape": [2, 3],
                        "dtype": "float16",
                        "distribution": "uniform",
                        "seed": 1,
                    },
                    "cache": {
                        "shape": [4, 3],
                        "dtype": "float16",
                        "distribution": "uniform",
                        "seed": 2,
                    },
                },
            }
        ],
    }
    contract = {
        "schema_version": 2,
        "cases": {
            "small": {
                "outputs": {
                    "updated": {"shape": [4, 3], "dtype": "float16"},
                    "indices": {"shape": [2], "dtype": "int32"},
                },
                "scalars": {"offset": {"kind": "int64", "value": 1}},
                "aliases": {"updated": "cache"},
            }
        },
    }
    return suite, contract


def test_contract_roundtrip_output_identity_and_seed_cases_share_native_domain():
    suite, contract = fixture()
    suite["cases"].append(copy.deepcopy(suite["cases"][0]))
    suite["cases"][1]["id"] = "other_seed"
    suite["cases"][1]["inputs"]["x"]["seed"] = 99
    contract["cases"]["other_seed"] = copy.deepcopy(contract["cases"]["small"])
    suite = RecipeSuiteV2.model_validate(suite)
    parsed = TensorContractV2.model_validate(contract)
    parsed.validate_suite(suite)
    assert parse_tensor_contract(parsed.canonical_bytes(), suite) == parsed
    assert len(parsed.sha256) == 64
    assert parsed.output_contract().cases["small"]["indices"].dtype == "int32"
    contract["cases"]["small"]["scalars"]["offset"]["value"] = 2
    changed = TensorContractV2.model_validate(contract)
    assert changed.sha256 != parsed.sha256


@pytest.mark.parametrize(
    "kind,value",
    [
        ("bool", 1),
        ("int64", True),
        ("int64", 2**63),
        ("int64", -(2**63) - 1),
        ("float64", 1),
        ("float64", float("nan")),
        ("float64", float("inf")),
        ("string", "x"),
    ],
)
def test_invalid_scalar_values(kind, value):
    with pytest.raises(ValueError):
        ScalarValueV2.model_validate({"kind": kind, "value": value})


def test_valid_scalar_limits():
    for kind, value in [
        ("int64", -(2**63)),
        ("int64", 2**63 - 1),
        ("bool", True),
        ("float64", -0.0),
    ]:
        assert ScalarValueV2.model_validate({"kind": kind, "value": value}).value == value


@pytest.mark.parametrize(
    "mutation",
    [
        lambda s, c: c["cases"].clear(),
        lambda s, c: c["cases"]["small"]["aliases"].update(unknown="cache"),
        lambda s, c: c["cases"]["small"]["aliases"].update(indices="cache"),
        lambda s, c: c["cases"]["small"]["aliases"].update(updated="unknown"),
        lambda s, c: c["cases"]["small"]["outputs"]["updated"].update(shape=[2, 3]),
        lambda s, c: c["cases"]["small"]["scalars"].update(x={"kind": "bool", "value": False}),
        lambda s, c: s["cases"][0]["inputs"]["x"].update(layout="transpose"),
    ],
)
def test_invalid_mutation_and_case_domains(mutation):
    suite, contract = fixture()
    mutation(suite, contract)
    with pytest.raises(ValueError):
        TensorContractV2.model_validate(contract).validate_suite(
            RecipeSuiteV2.model_validate(suite)
        )


def test_duplicate_native_domain_cannot_declare_different_outputs():
    suite, contract = fixture()
    suite["cases"].append(copy.deepcopy(suite["cases"][0]))
    suite["cases"][1]["id"] = "other"
    contract["cases"]["other"] = copy.deepcopy(contract["cases"]["small"])
    contract["cases"]["other"]["outputs"]["indices"]["shape"] = [3]
    with pytest.raises(ValueError, match="Ambiguous"):
        TensorContractV2.model_validate(contract).validate_suite(
            RecipeSuiteV2.model_validate(suite)
        )


@pytest.mark.parametrize(
    "dtype,code", [("int8", "b"), ("int32", "i"), ("int64", "q"), ("bool", "B")]
)
def test_integer_outputs_stay_exact_despite_loose_tolerance(dtype, code):
    high = 2**63 - 2 if dtype == "int64" else 0
    ref, got = struct.pack("<" + code, high), struct.pack("<" + code, high + 1)
    result = compare_tensor(ref, got, dtype, rtol=1e20, atol=1e20)
    assert result == {
        "metric": "exact",
        "passed": False,
        "elements": 1,
        "mismatches": 1,
        "max_absolute_error": 1,
    }
    assert compare_tensor(ref, ref, dtype, rtol=0, atol=0)["passed"]


@pytest.mark.parametrize(
    "dtype,ref,got",
    [
        ("float16", struct.pack("<e", 1), struct.pack("<e", 1.001)),
        ("bfloat16", bytes.fromhex("803f"), bytes.fromhex("813f")),
        ("float32", struct.pack("<f", 1), struct.pack("<f", 1.001)),
    ],
)
def test_float_storage_uses_tolerances(dtype, ref, got):
    assert compare_tensor(ref, got, dtype, rtol=0.01, atol=0)["passed"]
    assert not compare_tensor(ref, got, dtype, rtol=0, atol=0)["passed"]


def test_nonfinite_empty_and_invalid_payloads():
    nan = struct.pack("<e", float("nan"))
    assert not compare_tensor(nan, nan, "float16", rtol=0, atol=0)["passed"]
    assert compare_tensor(nan, nan, "float16", rtol=0, atol=0, equal_nan=True)["passed"]
    assert compare_tensor(b"", b"", "int64", rtol=0, atol=0)["passed"]
    for ref, got, dtype in [
        (b"x", b"x", "float16"),
        (b"", b"x", "int8"),
        (b"\x02", b"\x02", "bool"),
        (b"", b"", "float64"),
    ]:
        with pytest.raises(ValueError):
            compare_tensor(ref, got, dtype, rtol=0, atol=0)
    with pytest.raises(ValueError):
        compare_tensor(b"", b"", "int64", rtol=float("nan"), atol=0)
