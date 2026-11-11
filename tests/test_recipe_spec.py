import copy
import hashlib
import json

import pytest
import yaml
from spec_fixtures import spec_files

from cannery.registry.manifest import RecipeEntrypoint, SpecError, validate_spec


def recipe_spec():
    files = spec_files(reference=b"raise AssertionError('intake must not import author code')\n")
    manifest = yaml.safe_load(files["spec/cannery.yaml"])
    manifest["spec"]["cases"] = {"recipe_file": "recipes.yaml"}
    files["spec/cannery.yaml"] = yaml.safe_dump(manifest).encode()
    files.pop("spec/cases.py")
    recipes = {
        "schema_version": 1,
        "generator": "sha256-f32-v1",
        "cases": [
            {
                "id": "small",
                "inputs": {"x": {"shape": [2, 3], "distribution": "uniform_f32", "seed": 42}},
            },
            {
                "id": "strided",
                "inputs": {
                    "x": {
                        "shape": [4, 3],
                        "distribution": "arange_f32",
                        "seed": 42,
                        "layout": "transpose",
                    }
                },
            },
            {
                "id": "empty",
                "inputs": {"x": {"shape": [0, 3], "distribution": "zeros_f32", "seed": 42}},
            },
        ],
    }
    files["spec/recipes.yaml"] = yaml.safe_dump(recipes).encode()
    return files, manifest, recipes


def test_recipe_specs_validate_without_importing_python_and_bind_all_source():
    files, manifest, recipes = recipe_spec()
    spec = validate_spec(files, "spec")
    assert isinstance(spec.manifest.spec.cases, RecipeEntrypoint)
    manifest["spec"]["environment_lock"] = "environment.lock"
    manifest["spec"]["validation"]["equal_nan"] = False
    assert spec.manifest.model_dump(mode="json") == manifest
    assert "spec/recipes.yaml" in spec.file_sha256
    before = spec.content_sha256
    recipes["cases"][0]["inputs"]["x"]["seed"] = 43
    files["spec/recipes.yaml"] = yaml.safe_dump(recipes).encode()
    assert validate_spec(files, "spec").content_sha256 != before


def test_legacy_python_manifest_semantic_hash_stays_exact():
    files = spec_files()
    manifest = yaml.safe_load(files["spec/cannery.yaml"])
    manifest["spec"]["environment_lock"] = "environment.lock"
    manifest["spec"]["validation"]["equal_nan"] = False
    normalized = copy.deepcopy(manifest)
    normalized["kernel"].pop("description")
    normalized["spec"].pop("version")
    expected = hashlib.sha256(
        json.dumps(
            {
                "manifest": normalized,
                "files": {
                    name: hashlib.sha256(data).hexdigest()
                    for name, data in sorted(files.items())
                    if name != "spec/cannery.yaml"
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    validated = validate_spec(files, "spec")
    assert validated.content_sha256 == expected
    assert validated.manifest.model_dump(mode="json") == manifest


@pytest.mark.parametrize(
    "cases",
    [
        {"recipe_file": "../outside"},
        {"recipe_file": "/outside"},
        {"recipe_file": "recipes.yaml", "file": "cases.py", "function": "generate_cases"},
        {"recipe_file": "missing.yaml"},
    ],
)
def test_missing_ambiguous_or_unsafe_recipe_file_fails(cases):
    files, manifest, _ = recipe_spec()
    manifest["spec"]["cases"] = cases
    files["spec/cannery.yaml"] = yaml.safe_dump(manifest).encode()
    with pytest.raises(SpecError):
        validate_spec(files, "spec")


@pytest.mark.parametrize("change", ["name", "many_inputs", "many_outputs", "generator", "bad_seed"])
def test_recipe_interface_and_algorithm_are_checked_during_intake(change):
    files, manifest, recipes = recipe_spec()
    if change == "name":
        recipes["cases"][0]["inputs"] = {"other": recipes["cases"][0]["inputs"]["x"]}
    elif change == "many_inputs":
        manifest["spec"]["interface"]["inputs"] = ["x", "other"]
    elif change == "many_outputs":
        manifest["spec"]["interface"]["outputs"] = ["result", "other"]
    elif change == "generator":
        recipes["generator"] = "unknown"
    else:
        recipes["cases"][0]["inputs"]["x"]["seed"] = True
    files["spec/cannery.yaml"] = yaml.safe_dump(manifest).encode()
    files["spec/recipes.yaml"] = yaml.safe_dump(recipes).encode()
    with pytest.raises(SpecError):
        validate_spec(files, "spec")
