"""Frozen tensor spec reader; public registry admission remains disabled."""

import json
from dataclasses import dataclass

from cannery.git_intake import relative_source_path
from cannery.numerics.recipes_v2 import RecipeSuiteV2, parse_recipes_v2
from cannery.numerics.tensor_contract import TensorContractV2, parse_tensor_contract
from cannery.registry.manifest import (
    EnvironmentLock,
    SpecError,
    TensorRecipeEntrypoint,
    ValidatedSpec,
    _check_spec_paths,
    _freeze_spec,
    parse_manifest,
    unique_pairs,
)


@dataclass(frozen=True)
class ValidatedTensorSpec:
    spec: ValidatedSpec
    suite: RecipeSuiteV2
    contract: TensorContractV2


def validate_tensor_spec(files: dict[str, bytes], source_path: str) -> ValidatedTensorSpec:
    """Validate named recipes, outputs, scalars and mutation aliases without imports."""
    prefix = "" if source_path == "." else relative_source_path(source_path) + "/"
    try:
        manifest_path = prefix + "cannery.yaml"
        manifest = parse_manifest(files[manifest_path])
        cases = manifest.spec.cases
        if manifest.schema_version != 2 or not isinstance(cases, TensorRecipeEntrypoint):
            raise SpecError("Expected a tensor spec binding")
        required = [
            manifest.spec.reference.file,
            manifest.spec.environment_lock,
            cases.recipe_file,
            cases.tensor_contract_file,
        ]
        if any(prefix + name not in files for name in required):
            raise SpecError("Referenced tensor spec file is missing")
        environment = EnvironmentLock.model_validate(
            json.loads(
                files[prefix + manifest.spec.environment_lock],
                object_pairs_hook=unique_pairs,
            )
        )
        _check_spec_paths(files)
        suite = parse_recipes_v2(files[prefix + cases.recipe_file])
        contract = parse_tensor_contract(files[prefix + cases.tensor_contract_file], suite)
        interface = manifest.spec.interface
        if any(set(case.inputs) != set(interface.inputs) for case in suite.cases) or any(
            set(case.outputs) != set(interface.outputs) for case in contract.cases.values()
        ):
            raise SpecError("Tensor recipe/output names differ from the frozen interface")
        return ValidatedTensorSpec(
            _freeze_spec(files, manifest_path, manifest, environment),
            suite,
            contract,
        )
    except (KeyError, ValueError, TypeError, RecursionError):
        raise SpecError("Invalid or incomplete tensor specification files") from None
