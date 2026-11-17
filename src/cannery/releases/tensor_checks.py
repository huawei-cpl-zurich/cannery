"""Read-only publication checks for trusted named-output numerical attestations."""

import math
import re
from typing import Any

from cannery.numerics.comparison import Comparison
from cannery.registry.freeze import BaselineEvidence
from cannery.registry.tensor_manifest import ValidatedTensorSpec
from cannery.releases.contract import SHA, Tolerance, canonical
from cannery.releases.eligibility import require


def comparison(
    data: dict[str, Any], dtype: str, elements: int, pair: Tolerance, equal_nan: bool
) -> None:
    require(
        data["passed"] is True
        and type(data["elements"]) is int
        and data["elements"] == elements
        and type(data["mismatches"]) is int
        and data["mismatches"] == 0
    )
    if dtype not in {"float16", "bfloat16", "float32"}:
        require(
            set(data) == {"metric", "passed", "elements", "mismatches", "max_absolute_error"}
            and data["metric"] == "exact"
            and type(data["max_absolute_error"]) is int
            and data["max_absolute_error"] == 0
        )
        return
    require(data["metric"] == "torch_close")
    result = Comparison(**{key: value for key, value in data.items() if key != "metric"})
    require(
        result.rtol == pair.rtol
        and result.atol == pair.atol
        and result.equal_nan is equal_nan
        and result.nonfinite_mismatches == 0
    )
    for name in [
        "finite_pairs",
        "reference_nonfinite",
        "candidate_nonfinite",
        "nonfinite_mismatches",
        "zero_reference_mismatches",
    ]:
        require(type(data[name]) is int and 0 <= data[name] <= elements)
    require(
        result.finite_pairs + result.reference_nonfinite == elements
        and result.reference_nonfinite == result.candidate_nonfinite
        and result.zero_reference_mismatches <= result.finite_pairs
    )
    for name in ["max_absolute_error", "max_relative_error", "rms_error"]:
        value = data[name]
        require(
            value is None
            and name == "max_relative_error"
            or type(value) in (int, float)
            and math.isfinite(value)
            and value >= 0
        )
    require((result.max_relative_error is None) == (result.zero_reference_mismatches > 0))


def check_tensor(
    report: dict[str, Any],
    evidence: BaselineEvidence,
    frozen: ValidatedTensorSpec,
    default: Tolerance,
    requested: Tolerance,
) -> None:
    """Verify complete frozen domains, both tolerances, exact indices and mutations.

    These are immutable controller attestations. Raw private oracle/candidate
    bytes are checked by numerical/benchmark execution, never exposed here.
    """
    suite, contract = frozen.suite, frozen.contract
    contract.validate_suite(suite)
    require(
        report["schema_version"] == 2
        and report["passed"] is True
        and report["generator"] == evidence.generator == suite.generator
        and report["baseline_key"] == evidence.baseline_key
        and report["spec_sha256"] == evidence.spec_sha256 == frozen.spec.content_sha256
        and report["suite_sha256"] == evidence.suite_sha256 == suite.sha256
    )
    rows = report["cases"]
    require(
        len(rows) == evidence.cases == len(suite.cases)
        and [row["id"] for row in rows] == [case.id for case in suite.cases]
    )
    for row, recipe in zip(rows, suite.cases, strict=True):
        definition = contract.cases[recipe.id]
        require(
            row["passed"] is True
            and row["readonly_inputs_unchanged"] is True
            and row["mutable_aliases_consistent"] is True
        )
        require(
            canonical(row["scalars"])
            == canonical(
                {
                    name: scalar.model_dump(mode="json")
                    for name, scalar in definition.scalars.items()
                }
            )
        )
        require(
            set(row["inputs"]) == set(recipe.inputs)
            and set(row["outputs"]) == set(definition.outputs)
        )
        mutable = set(definition.aliases.values())
        for name, state in row["inputs"].items():
            require(
                set(state) == {"mutable", "sha256", "unchanged"}
                and state["mutable"] is (name in mutable)
                and type(state["unchanged"]) is bool
                and type(state["sha256"]) is str
                and re.fullmatch(SHA, state["sha256"])
            )
            require(name in mutable or state["unchanged"] is True)
        for name, descriptor in definition.outputs.items():
            output = row["outputs"][name]
            require(
                set(output)
                == {"shape", "dtype", "alias", "passed", "spec_default", "submission_requested"}
                and output["passed"] is True
                and canonical(output["shape"]) == canonical(list(descriptor.shape))
                and output["dtype"] == descriptor.dtype
                and output["alias"] == definition.aliases.get(name)
            )
            for key, pair in [("spec_default", default), ("submission_requested", requested)]:
                comparison(
                    output[key],
                    descriptor.dtype,
                    math.prod(descriptor.shape),
                    pair,
                    frozen.spec.manifest.spec.validation.equal_nan,
                )
