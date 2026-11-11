"""Evaluate candidate tensors against cached outputs in the trusted controller."""

from typing import Any

from cannery.numerics.baseline_format import BaselineBundle, records
from cannery.numerics.comparison import compare
from cannery.numerics.recipes import RecipeSuite
from cannery.registry.manifest import Validation
from cannery.submissions.manifest import RequestedValidation


def evaluate(
    baseline: BaselineBundle,
    suite: RecipeSuite,
    outputs: dict[str, bytes],
    input_hashes: dict[str, str],
    defaults: Validation,
    requested: RequestedValidation,
) -> dict[str, Any]:
    """Require both declared tolerance pairs and unchanged inputs for every case."""
    identifiers = {c.id for c in suite.cases}
    if set(outputs) != identifiers or set(input_hashes) != identifiers:
        raise ValueError("Candidate results must cover every mandatory case exactly once")
    expected = records(suite, baseline.outputs)
    if (
        baseline.metadata.get("suite_sha256") != suite.sha256
        or baseline.metadata.get("generator") != suite.generator
        or baseline.metadata.get("cases") != expected
    ):
        raise ValueError("Baseline descriptors differ from the mandatory suite")
    cases = []
    for case, descriptor in zip(suite.cases, expected, strict=True):
        reference, actual = baseline.outputs[case.id], outputs[case.id]
        unchanged = input_hashes[case.id] == descriptor["input_sha256"]
        spec_result = compare(
            reference,
            actual,
            rtol=defaults.default_rtol,
            atol=defaults.default_atol,
            equal_nan=defaults.equal_nan,
        )
        submission_result = compare(
            reference,
            actual,
            rtol=requested.requested_rtol,
            atol=requested.requested_atol,
            equal_nan=defaults.equal_nan,
        )
        cases.append(
            {
                "id": case.id,
                "shape": descriptor["shape"],
                "dtype": "float32",
                "input_unchanged": unchanged,
                "spec_default": spec_result.as_dict(),
                "submission_requested": submission_result.as_dict(),
                "passed": unchanged and spec_result.passed and submission_result.passed,
            }
        )
    return {
        "schema_version": 1,
        "baseline_key": baseline.metadata["key"],
        "spec_sha256": baseline.metadata["spec_sha256"],
        "suite_sha256": suite.sha256,
        "generator": suite.generator,
        "passed": all(case["passed"] for case in cases),
        "cases": cases,
    }
