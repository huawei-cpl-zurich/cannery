"""Versioned synchronized-call timing and trusted-controller score calculations."""

import json
import math
import statistics
from collections.abc import Callable
from typing import Any, Literal

from pydantic import Field, model_validator

from cannery.artifacts import digest
from cannery.numerics.recipes import RecipeSuite
from cannery.registry.manifest import StrictModel, unique_pairs

PROTOCOL: Literal["sync-call-v1"] = "sync-call-v1"
WARMUP = 20
SAMPLES = 100
SHA = r"^[a-f0-9]{64}$"


class Cohort(StrictModel):
    """Trusted worker identity; device identity must come from hardware discovery."""

    device: Literal["a5"] = "a5"
    hardware_id: str = Field(min_length=1, max_length=256)
    driver: str = Field(min_length=1, max_length=256)
    image: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    loader_sha256: str = Field(pattern=SHA)
    runtime: dict[str, str]
    protocol: Literal["sync-call-v1"] = PROTOCOL

    @model_validator(mode="after")
    def complete_runtime(self) -> "Cohort":
        if set(self.runtime) != {
            "machine",
            "python",
            "torch",
            "torch_npu",
            "cann",
            "soc",
        } or not all(self.runtime.values()):
            raise ValueError("Incomplete runtime fingerprint")
        return self


def cache_key(spec_sha256: str, suite: RecipeSuite, cohort: Cohort) -> str:
    if len(spec_sha256) != 64 or any(c not in "0123456789abcdef" for c in spec_sha256):
        raise ValueError("Invalid specification digest")
    return digest(
        json.dumps(
            {
                "spec_sha256": spec_sha256,
                "suite_sha256": suite.sha256,
                "generator": suite.generator,
                "cohort": cohort.model_dump(mode="json"),
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    )


def measure(
    forward: Callable[[Any], Any],
    x: Any,
    synchronize: Callable[[], None],
    clock_ns: Callable[[], int],
    *,
    snapshot: Callable[[Any], Any] | None = None,
) -> tuple[list[float], Any, Any]:
    """Caller owns no_grad, input preparation and output verification outside timing."""
    before = forward(x)
    synchronize()
    if snapshot is not None:
        before = snapshot(before)
    for _ in range(WARMUP):
        output = forward(x)
        synchronize()
        del output
    samples = []
    for _ in range(SAMPLES):
        synchronize()
        start = clock_ns()
        output = forward(x)
        synchronize()
        samples.append((clock_ns() - start) / 1000)
        del output  # Destruction is outside the measured interval.
    after = forward(x)
    synchronize()
    if snapshot is not None:
        after = snapshot(after)
    return samples, before, after


def parse_samples(raw: bytes, suite: RecipeSuite) -> dict[str, list[float]]:
    """Only timings cross this wire; tensor evidence uses the numerical protocol."""
    if type(raw) is not bytes or len(raw) > 256 * 1024:
        raise ValueError("Invalid timing output size")
    try:
        payload = json.loads(raw, object_pairs_hook=unique_pairs)
        expected = [c.id for c in suite.cases if math.prod(next(iter(c.inputs.values())).shape)]
        if (
            set(payload) != {"protocol", "warmup", "cases"}
            or payload["protocol"] != PROTOCOL
            or type(payload["warmup"]) is not int
            or payload["warmup"] != WARMUP
            or not isinstance(payload["cases"], list)
            or [c["id"] for c in payload["cases"]] != expected
        ):
            raise ValueError
        result = {}
        for case in payload["cases"]:
            if set(case) != {"id", "samples_us"}:
                raise ValueError
            summarize(case["samples_us"])
            result[case["id"]] = [float(v) for v in case["samples_us"]]
        return result
    except (ValueError, KeyError, TypeError, OverflowError, RecursionError):
        raise ValueError("Invalid timing output") from None


def summarize(samples: list[float]) -> dict[str, float | int]:
    try:
        valid = (
            type(samples) is list
            and len(samples) == SAMPLES
            and all(type(v) in (float, int) and math.isfinite(v) and v > 0 for v in samples)
        )
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError("Expected 100 finite positive timing samples")
    ordered = sorted(samples)
    return {
        "median_us": ordered[49] + (ordered[50] - ordered[49]) / 2,
        "p95_us": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "samples": len(ordered),
    }


def score(
    suite: RecipeSuite,
    reference: dict[str, list[float]],
    candidate: dict[str, list[float]],
) -> dict[str, Any]:
    expected = {c.id for c in suite.cases if math.prod(next(iter(c.inputs.values())).shape)}
    if set(reference) != expected or set(candidate) != expected:
        raise ValueError("Timing must cover every nonempty mandatory case")
    cases, ratios = [], []
    for case in suite.cases:
        shape = next(iter(case.inputs.values())).shape
        row: dict[str, Any] = {"id": case.id, "shape": list(shape)}
        if not math.prod(shape):
            row.update(excluded="empty_correctness_only")
        else:
            baseline, actual = summarize(reference[case.id]), summarize(candidate[case.id])
            ratio = baseline["median_us"] / actual["median_us"]
            if not math.isfinite(ratio) or ratio <= 0:
                raise ValueError("Unrepresentable speedup")
            ratios.append(ratio)
            row.update(reference=baseline, candidate=actual, speedup=ratio)
        cases.append(row)
    geomean = math.exp(statistics.mean(math.log(r) for r in ratios)) if ratios else None
    return {"protocol": PROTOCOL, "cases": cases, "geomean_speedup": geomean}
