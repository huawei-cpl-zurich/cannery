import copy
import hashlib
import json
import struct

import pytest
from test_baseline_v2 import fixture as bundle_fixture
from test_baseline_v2 import pack as pack_fixture
from test_recipe_spec import recipe_spec
from test_tensor_numerical_contract import fixture as contract_fixture

from cannery.numerics.baseline_v2 import pack_baseline_v2, unpack_baseline_v2
from cannery.numerics.evaluation_v2 import evaluate_tensor
from cannery.numerics.recipes_v2 import RecipeSuiteV2, input_sha256_v2
from cannery.numerics.tensor_contract import TensorContractV2
from cannery.registry.manifest import Validation, validate_spec
from cannery.submissions.manifest import RequestedValidation


@pytest.fixture
def domain():
    recipes, definition = contract_fixture()
    definition["cases"]["small"]["outputs"]["indices"]["dtype"] = "int64"
    definition["cases"]["small"]["outputs"]["scale"] = {"shape": [1], "dtype": "float32"}
    suite = RecipeSuiteV2.model_validate(recipes)
    contract = TensorContractV2.model_validate(definition)
    files, _, _ = recipe_spec()
    environment = validate_spec(files, "spec").environment
    outputs = {
        "small": {
            "updated": struct.pack("<12e", *range(12)),
            "indices": struct.pack("<2q", 2**62, 2**62 + 1),
            "scale": struct.pack("<f", 1.0),
        }
    }
    package = pack_baseline_v2(
        "b" * 64,
        suite,
        environment,
        "sha256:" + "a" * 64,
        contract.output_contract(),
        outputs,
    )
    baseline = unpack_baseline_v2(
        package,
        hashlib.sha256(package).hexdigest(),
        spec_sha256="b" * 64,
        suite=suite,
        environment=environment,
        contract=contract.output_contract(),
    )
    hashes = {"small": dict(baseline.metadata["cases"][0]["inputs"])}
    hashes["small"]["cache"] = input_sha256_v2(
        suite.cases[0].inputs["cache"], outputs["small"]["updated"]
    )
    return baseline, suite, contract, copy.deepcopy(outputs), hashes


def evaluate(domain, default=0.0, requested=0.0):
    return evaluate_tensor(
        *domain,
        Validation(metric="torch_close", default_rtol=0, default_atol=default),
        RequestedValidation(requested_rtol=0, requested_atol=requested),
    )


def test_bundle_refactor_preserves_existing_bytes():
    assert hashlib.sha256(pack_fixture(*bundle_fixture())).hexdigest() == (
        "f685e52a7dde77d0ca01463c17e6725807f4e29f306d1b7eb41b541861f77b6c"
    )


def test_named_outputs_scalar_and_declared_mutation(domain):
    updated_hash = domain[4]["small"]["cache"]
    result = evaluate(domain)
    assert result["passed"] and result["schema_version"] == 2
    case = result["cases"][0]
    assert case["inputs"]["cache"] == {
        "mutable": True,
        "sha256": updated_hash,
        "unchanged": False,
    }
    assert case["readonly_inputs_unchanged"]
    assert case["mutable_aliases_consistent"]
    assert case["outputs"]["updated"]["alias"] == "cache"
    assert case["scalars"]["offset"] == {"kind": "int64", "value": 1}
    json.dumps(result, allow_nan=False)


def test_mutable_input_hash_must_match_returned_alias_bytes(domain):
    domain[4]["small"]["cache"] = "0" * 64
    result = evaluate(domain)
    assert not result["passed"]
    assert not result["cases"][0]["mutable_aliases_consistent"]
    assert all(output["passed"] for output in result["cases"][0]["outputs"].values())


def test_readonly_input_mutation_fails_even_with_correct_outputs(domain):
    domain[4]["small"]["x"] = "0" * 64
    case = evaluate(domain)["cases"][0]
    assert not case["passed"] and not case["readonly_inputs_unchanged"]
    assert all(output["passed"] for output in case["outputs"].values())


@pytest.mark.parametrize("default,requested", [(0.0, 1.0), (1.0, 0.0)])
def test_both_tolerance_pairs_required(domain, default, requested):
    domain[3]["small"]["scale"] = struct.pack("<f", 1.25)
    result = evaluate(domain, default, requested)
    assert not result["passed"]
    scale = result["cases"][0]["outputs"]["scale"]
    assert scale["spec_default"]["passed"] == bool(default)
    assert scale["submission_requested"]["passed"] == bool(requested)
    assert evaluate(domain, 1.0, 1.0)["passed"]


def test_int64_low_bit_mismatch_never_uses_float_tolerance(domain):
    domain[3]["small"]["indices"] = struct.pack("<2q", 2**62, 2**62)
    result = evaluate(domain, 1e20, 1e20)
    assert not result["passed"]
    comparison = result["cases"][0]["outputs"]["indices"]["spec_default"]
    assert comparison["metric"] == "exact" and comparison["mismatches"] == 1


@pytest.mark.parametrize("target", ["cases", "outputs", "inputs"])
@pytest.mark.parametrize("extra", [False, True])
def test_missing_extra_names_rejected(domain, target, extra):
    values = {"cases": domain[3], "outputs": domain[3]["small"], "inputs": domain[4]["small"]}[
        target
    ]
    if extra:
        values["unexpected"] = next(iter(values.values()))
    else:
        values.pop(next(iter(values)))
    with pytest.raises(ValueError):
        evaluate(domain)


@pytest.mark.parametrize("value", ["bad", "A" * 64, None])
def test_invalid_hash_rejected(domain, value):
    domain[4]["small"]["cache"] = value
    with pytest.raises(ValueError):
        evaluate(domain)


def test_truncated_output_rejected(domain):
    domain[3]["small"]["updated"] = b""
    with pytest.raises(ValueError):
        evaluate(domain)


@pytest.mark.parametrize(
    "key,value",
    [
        ("schema_version", 1),
        ("suite_sha256", "0" * 64),
        ("generator", "sha256-uniform-v1"),
        ("cases", []),
        ("key", None),
        ("spec_sha256", "invalid"),
    ],
)
def test_mismatched_baseline_rejected(domain, key, value):
    domain[0].metadata[key] = value
    with pytest.raises(ValueError):
        evaluate(domain)
