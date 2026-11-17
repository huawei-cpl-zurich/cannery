"""Trusted shared-case scoring for named tensor implementations."""

import math
import statistics
from typing import Any

from cannery.benchmark.contract import summarize
from cannery.benchmark.tensor_timing import TENSOR_PROTOCOL, tensor_timed_cases
from cannery.numerics.recipes_v2 import RecipeSuiteV2
from cannery.numerics.tensor_contract import TensorContractV2


def tensor_score(
    suite: RecipeSuiteV2,
    contract: TensorContractV2,
    reference: dict[str, list[float]],
    candidate: dict[str, list[float]],
    *,
    first_input: str,
) -> dict[str, Any]:
    if any(first_input not in case.inputs for case in suite.cases):
        raise ValueError("Display input must belong to every frozen case")
    timed = set(tensor_timed_cases(suite, contract))
    if set(reference) != timed or set(candidate) != timed:
        raise ValueError("Timing must cover every nonempty tensor case")
    cases, ratios = [], []
    for case in suite.cases:
        row: dict[str, Any] = {"id": case.id, "shape": list(case.inputs[first_input].shape)}
        if case.id not in timed:
            row["excluded"] = "empty_correctness_only"
        else:
            baseline, actual = summarize(reference[case.id]), summarize(candidate[case.id])
            ratio = baseline["median_us"] / actual["median_us"]
            if not math.isfinite(ratio) or ratio <= 0:
                raise ValueError("Unrepresentable tensor speedup")
            ratios.append(ratio)
            row.update(reference=baseline, candidate=actual, speedup=ratio)
        cases.append(row)
    mean = math.exp(statistics.mean(math.log(r) for r in ratios)) if ratios else None
    return {"protocol": TENSOR_PROTOCOL, "cases": cases, "geomean_speedup": mean}
