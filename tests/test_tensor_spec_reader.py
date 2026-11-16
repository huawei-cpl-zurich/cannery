import json

import pytest
import yaml
from test_recipe_spec import recipe_spec
from test_tensor_numerical_contract import fixture

from cannery.registry.manifest import SpecError, parse_manifest, validate_spec
from cannery.registry.tensor_manifest import validate_tensor_spec


def tensor_files():
    files, _, _ = recipe_spec()
    manifest = yaml.safe_load(files["spec/cannery.yaml"])
    manifest["schema_version"] = 2
    manifest["spec"]["cases"] = {
        "recipe_file": "recipes.yaml",
        "tensor_contract_file": "tensor-contract.yaml",
    }
    manifest["spec"]["interface"] = {"inputs": ["x", "cache"], "outputs": ["updated", "indices"]}
    suite, contract = fixture()
    files["spec/cannery.yaml"] = yaml.safe_dump(manifest).encode()
    files["spec/recipes.yaml"] = json.dumps(suite).encode()
    files["spec/tensor-contract.yaml"] = json.dumps(contract).encode()
    files["spec/reference.py"] = b"raise RuntimeError('must never import during spec intake')\n"
    return files


def test_reader_freezes_full_domain_without_importing_reference_and_admission_stays_disabled():
    files = tensor_files()
    frozen = validate_tensor_spec(files, "spec")
    assert frozen.spec.manifest.schema_version == 2
    assert frozen.contract.cases["small"].aliases == {"updated": "cache"}
    assert frozen.suite.cases[0].inputs["x"].dtype == "float16"
    dumped = frozen.spec.manifest.model_dump(mode="json")
    assert dumped["spec"]["cases"]["tensor_contract_file"] == "tensor-contract.yaml"
    with pytest.raises(SpecError):
        validate_spec(files, "spec")
    root_files = {name.removeprefix("spec/"): data for name, data in files.items()}
    assert validate_tensor_spec(root_files, ".").contract == frozen.contract


def test_existing_frozen_identity_remains_byte_compatible():
    files, _, _ = recipe_spec()
    assert validate_spec(files, "spec").content_sha256 == (
        "824183e657c0acfa91d575bf0d1204bf89ddd220a94e4a46419175eb6788258e"
    )
    with pytest.raises(SpecError):
        validate_tensor_spec(files, "spec")


@pytest.mark.parametrize(
    "path",
    [
        "spec/reference.py",
        "spec/recipes.yaml",
        "spec/tensor-contract.yaml",
        "spec/environment.lock",
    ],
)
def test_missing_frozen_components_rejected(path):
    files = tensor_files()
    del files[path]
    with pytest.raises(SpecError):
        validate_tensor_spec(files, "spec")


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": 1},
        {"cases": {"recipe_file": "recipes.yaml"}},
        {"cases": {"recipe_file": "recipes.yaml", "tensor_contract_file": "../escape"}},
        {"interface": {"inputs": ["wrong"], "outputs": ["updated", "indices"]}},
        {"interface": {"inputs": ["x", "cache"], "outputs": ["wrong"]}},
    ],
)
def test_invalid_binding_and_interface_rejected(change):
    files = tensor_files()
    manifest = yaml.safe_load(files["spec/cannery.yaml"])
    for key, value in change.items():
        if key == "schema_version":
            manifest[key] = value
        else:
            manifest["spec"][key] = value
    files["spec/cannery.yaml"] = yaml.safe_dump(manifest).encode()
    with pytest.raises(SpecError):
        validate_tensor_spec(files, "spec")


def test_case_scalars_and_mutation_policy_change_identity():
    files = tensor_files()
    before = validate_tensor_spec(files, "spec").spec.content_sha256
    contract = json.loads(files["spec/tensor-contract.yaml"])
    contract["cases"]["small"]["scalars"]["offset"]["value"] = 2
    files["spec/tensor-contract.yaml"] = json.dumps(contract).encode()
    assert validate_tensor_spec(files, "spec").spec.content_sha256 != before
    contract["cases"]["small"]["scalars"]["offset"]["value"] = 1
    contract["cases"]["small"]["aliases"] = {}
    files["spec/tensor-contract.yaml"] = json.dumps(contract).encode()
    assert validate_tensor_spec(files, "spec").spec.content_sha256 != before


def test_source_paths_and_duplicate_environment_fields_rejected():
    for name in ["spec/../escape", "spec/.git/config"]:
        files = tensor_files()
        files[name] = b"forbidden"
        with pytest.raises(SpecError):
            validate_tensor_spec(files, "spec")
    files = tensor_files()
    files["spec/environment.lock"] = (
        b'{"schema_version":1,"python":"3.12","python":"3.13","torch":"2.6.0"}'
    )
    with pytest.raises(SpecError):
        validate_tensor_spec(files, "spec")
    manifest = yaml.safe_load(tensor_files()["spec/cannery.yaml"])
    manifest["schema_version"] = 1
    with pytest.raises(SpecError):
        parse_manifest(yaml.safe_dump(manifest).encode())
