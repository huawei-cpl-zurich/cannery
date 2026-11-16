"""Compare named outputs and enforce declared mutation in the trusted controller."""

import re
from typing import Any

from cannery.numerics.baseline_v2 import BaselineBundleV2, records_v2
from cannery.numerics.comparison_v2 import compare_tensor
from cannery.numerics.recipes_v2 import RecipeSuiteV2
from cannery.numerics.tensor_contract import TensorContractV2
from cannery.registry.manifest import Validation
from cannery.submissions.manifest import RequestedValidation


def evaluate_tensor(
    baseline: BaselineBundleV2,
    suite: RecipeSuiteV2,
    contract: TensorContractV2,
    outputs: dict[str, dict[str, bytes]],
    input_hashes: dict[str, dict[str, str]],
    defaults: Validation,
    requested: RequestedValidation,
) -> dict[str, Any]:
    """Require both tolerance pairs, exact integer outputs and unchanged read-only inputs."""
    contract.validate_suite(suite)
    identifiers = {case.id for case in suite.cases}
    if set(outputs) != identifiers or set(input_hashes) != identifiers:
        raise ValueError("Candidate must cover every mandatory tensor case exactly once")
    expected = records_v2(suite, contract.output_contract(), baseline.outputs)
    metadata = baseline.metadata
    if (
        metadata.get("schema_version") != 2
        or metadata.get("suite_sha256") != suite.sha256
        or metadata.get("generator") != suite.generator
        or metadata.get("cases") != expected
        or any(
            not isinstance(metadata.get(key), str)
            or re.fullmatch(r"[a-f0-9]{64}", metadata[key]) is None
            for key in ("key", "spec_sha256")
        )
    ):
        raise ValueError("Baseline differs from the mandatory tensor domain")
    cases = []
    for case, descriptor in zip(suite.cases, expected, strict=True):
        definition = contract.cases[case.id]
        actual_hashes = input_hashes[case.id]
        if (
            set(actual_hashes) != set(case.inputs)
            or set(outputs[case.id]) != set(definition.outputs)
            or any(
                not isinstance(h, str) or re.fullmatch(r"[a-f0-9]{64}", h) is None
                for h in actual_hashes.values()
            )
        ):
            raise ValueError("Candidate tensor names or input hashes differ from the contract")
        mutable = set(definition.aliases.values())
        inputs = {
            name: {
                "mutable": name in mutable,
                "sha256": actual_hashes[name],
                "unchanged": actual_hashes[name] == descriptor["inputs"][name],
            }
            for name in case.inputs
        }
        unchanged = all(state["mutable"] or state["unchanged"] for state in inputs.values())
        compared = {}
        for name, tensor in definition.outputs.items():
            reference, actual = baseline.outputs[case.id][name], outputs[case.id][name]
            spec_result = compare_tensor(
                reference,
                actual,
                tensor.dtype,
                rtol=defaults.default_rtol,
                atol=defaults.default_atol,
                equal_nan=defaults.equal_nan,
            )
            requested_result = compare_tensor(
                reference,
                actual,
                tensor.dtype,
                rtol=requested.requested_rtol,
                atol=requested.requested_atol,
                equal_nan=defaults.equal_nan,
            )
            compared[name] = {
                "shape": list(tensor.shape),
                "dtype": tensor.dtype,
                "alias": definition.aliases.get(name),
                "spec_default": spec_result,
                "submission_requested": requested_result,
                "passed": spec_result["passed"] and requested_result["passed"],
            }
        cases.append(
            {
                "id": case.id,
                "inputs": inputs,
                "outputs": compared,
                "scalars": {
                    name: s.model_dump(mode="json") for name, s in definition.scalars.items()
                },
                "readonly_inputs_unchanged": unchanged,
                "passed": unchanged and all(value["passed"] for value in compared.values()),
            }
        )
    return {
        "schema_version": 2,
        "baseline_key": metadata["key"],
        "spec_sha256": metadata["spec_sha256"],
        "suite_sha256": suite.sha256,
        "generator": suite.generator,
        "cases": cases,
        "passed": all(case["passed"] for case in cases),
    }
