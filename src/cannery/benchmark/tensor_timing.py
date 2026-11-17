"""Named-tensor timing with mutable-input reset excluded from every measurement."""

import json
from collections.abc import Callable
from typing import Any, Literal

from cannery.benchmark.contract import SAMPLES, WARMUP, summarize
from cannery.numerics.recipes_v2 import RecipeSuiteV2
from cannery.numerics.tensor_contract import TensorContractV2
from cannery.registry.manifest import unique_pairs

TENSOR_PROTOCOL: Literal["sync-call-tensor-v2"] = "sync-call-tensor-v2"


def measure_tensor(
    forward: Callable[[], Any],
    reset: Callable[[], None],
    synchronize: Callable[[], None],
    clock_ns: Callable[[], int],
    *,
    snapshot: Callable[[Any], Any],
) -> tuple[list[float], Any, Any]:
    """Reset aliased inputs, then synchronize before starting each call's timer.

    Caller owns no_grad, immutable original buffers and metadata checks. Snapshot
    must preserve output bytes before the next reset changes any returned alias.
    Reset/device/snapshot failures propagate; no samples are retried or omitted.
    """
    reset()
    synchronize()
    output = forward()
    synchronize()
    before = snapshot(output)
    del output
    for _ in range(WARMUP):
        reset()
        synchronize()
        output = forward()
        synchronize()
        del output
    samples = []
    for _ in range(SAMPLES):
        reset()
        synchronize()
        start = clock_ns()
        output = forward()
        synchronize()
        samples.append((clock_ns() - start) / 1000)
        del output
    reset()
    synchronize()
    output = forward()
    synchronize()
    after = snapshot(output)
    return samples, before, after


def tensor_timed_cases(suite: RecipeSuiteV2, contract: TensorContractV2) -> list[str]:
    """Only entirely empty call domains are correctness-only cases."""
    contract.validate_suite(suite)
    return [
        case.id
        for case in suite.cases
        if any(t.nbytes for t in case.inputs.values())
        or any(t.nbytes for t in contract.cases[case.id].outputs.values())
    ]


def parse_tensor_samples(
    raw: bytes, suite: RecipeSuiteV2, contract: TensorContractV2
) -> dict[str, list[float]]:
    if type(raw) is not bytes or len(raw) > 256 * 1024:
        raise ValueError("Invalid tensor timing output size")
    try:
        payload = json.loads(raw, object_pairs_hook=unique_pairs)
        if (
            set(payload) != {"protocol", "warmup", "cases"}
            or payload["protocol"] != TENSOR_PROTOCOL
            or type(payload["warmup"]) is not int
            or payload["warmup"] != WARMUP
            or not isinstance(payload["cases"], list)
            or [c["id"] for c in payload["cases"]] != tensor_timed_cases(suite, contract)
        ):
            raise ValueError
        result = {}
        for case in payload["cases"]:
            if set(case) != {"id", "samples_us"}:
                raise ValueError
            summarize(case["samples_us"])
            result[case["id"]] = [float(value) for value in case["samples_us"]]
        return result
    except (ValueError, KeyError, TypeError, OverflowError, RecursionError):
        raise ValueError("Invalid tensor timing output") from None
