import copy

import yaml
from test_tensor_spec_reader import tensor_files

from cannery.builds.tensor_signatures import tensor_variants
from cannery.registry.tensor_manifest import validate_tensor_spec


def test_variants_follow_spec_argument_order_and_preserve_aliases():
    files = tensor_files()
    manifest = yaml.safe_load(files["spec/cannery.yaml"])
    manifest["spec"]["interface"]["inputs"].reverse()
    manifest["spec"]["interface"]["outputs"].reverse()
    files["spec/cannery.yaml"] = yaml.safe_dump(manifest).encode()
    frozen = validate_tensor_spec(files, "spec")
    variants = tensor_variants(frozen)
    assert len(variants) == len(frozen.suite.cases)
    for variant in variants:
        assert [v["name"] for v in variant["inputs"]] == manifest["spec"]["interface"]["inputs"]
        assert [v["name"] for v in variant["outputs"]] == manifest["spec"]["interface"]["outputs"]
        definition = next(iter(frozen.contract.cases.values()))
        assert {v["name"]: v["alias"] for v in variant["outputs"]} == {
            name: definition.aliases.get(name) for name in definition.outputs
        }


def test_seed_only_cases_collapse_but_scalar_values_select_distinct_variant():
    files = tensor_files()
    recipes = yaml.safe_load(files["spec/recipes.yaml"])
    contract = yaml.safe_load(files["spec/tensor-contract.yaml"])
    duplicate = copy.deepcopy(recipes["cases"][0])
    duplicate["id"] = "new_seed"
    for recipe in duplicate["inputs"].values():
        recipe["seed"] += 1
    recipes["cases"].append(duplicate)
    contract["cases"]["new_seed"] = copy.deepcopy(contract["cases"][recipes["cases"][0]["id"]])
    files["spec/recipes.yaml"] = yaml.safe_dump(recipes).encode()
    files["spec/tensor-contract.yaml"] = yaml.safe_dump(contract).encode()
    frozen = validate_tensor_spec(files, "spec")
    original = tensor_variants(frozen)
    assert len(original) == len(recipes["cases"]) - 1
    contract["cases"]["new_seed"]["scalars"]["offset"]["value"] += 1
    files["spec/tensor-contract.yaml"] = yaml.safe_dump(contract).encode()
    changed = tensor_variants(validate_tensor_spec(files, "spec"))
    assert len(changed) == len(original) + 1


def test_float64_signed_zero_dispatch_bits_are_preserved():
    import struct

    files = tensor_files()
    recipes = yaml.safe_load(files["spec/recipes.yaml"])
    contract = yaml.safe_load(files["spec/tensor-contract.yaml"])
    duplicate = copy.deepcopy(recipes["cases"][0])
    duplicate["id"] = "negative_zero"
    recipes["cases"].append(duplicate)
    first = next(iter(contract["cases"]))
    contract["cases"][first]["scalars"] = {"offset": {"kind": "float64", "value": 0.0}}
    contract["cases"]["negative_zero"] = copy.deepcopy(contract["cases"][first])
    contract["cases"]["negative_zero"]["scalars"]["offset"]["value"] = -0.0
    files["spec/recipes.yaml"] = yaml.safe_dump(recipes).encode()
    files["spec/tensor-contract.yaml"] = yaml.safe_dump(contract).encode()
    variants = tensor_variants(validate_tensor_spec(files, "spec"))
    assert len(variants) == 2
    assert [struct.pack("<d", v["scalars"][0]["value"]).hex() for v in variants] == [
        "0000000000000000",
        "0000000000000080",
    ]
