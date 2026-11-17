"""Output-free tensor timing evidence, bound to the common measurement cohort."""

import json

from cannery.artifacts import ArtifactRef, verified
from cannery.benchmark.contract import WARMUP, Cohort, cache_key
from cannery.benchmark.tensor_timing import TENSOR_PROTOCOL, parse_tensor_samples
from cannery.numerics.recipes_v2 import RecipeSuiteV2
from cannery.numerics.tensor_contract import TensorContractV2
from cannery.registry.manifest import unique_pairs


def tensor_timing_artifact(
    spec_sha256: str,
    suite: RecipeSuiteV2,
    contract: TensorContractV2,
    cohort: Cohort,
    samples: dict[str, list[float]],
) -> bytes:
    if cohort.protocol != TENSOR_PROTOCOL:
        raise ValueError("Tensor timing cohort required")
    timing = {
        "protocol": TENSOR_PROTOCOL,
        "warmup": WARMUP,
        "cases": [
            {"id": case.id, "samples_us": samples[case.id]}
            for case in suite.cases
            if case.id in samples
        ],
    }
    checked = parse_tensor_samples(json.dumps(timing, allow_nan=False).encode(), suite, contract)
    if set(checked) != set(samples):
        raise ValueError("Unexpected tensor timing cases")
    return json.dumps(
        {"key": cache_key(spec_sha256, suite, cohort), "timing": timing},
        sort_keys=True,
        allow_nan=False,
    ).encode()


def read_tensor_timings(
    raw: bytes,
    artifact: ArtifactRef,
    spec_sha256: str,
    suite: RecipeSuiteV2,
    contract: TensorContractV2,
    cohort: Cohort,
) -> dict[str, list[float]]:
    verified(raw, artifact)
    if len(raw) > 256 * 1024 or cohort.protocol != TENSOR_PROTOCOL:
        raise ValueError("Invalid tensor timing evidence")
    try:
        value = json.loads(raw, object_pairs_hook=unique_pairs)
        if (
            not isinstance(value, dict)
            or set(value) != {"key", "timing"}
            or value["key"] != cache_key(spec_sha256, suite, cohort)
        ):
            raise ValueError
        return parse_tensor_samples(json.dumps(value["timing"]).encode(), suite, contract)
    except (ValueError, TypeError, KeyError, RecursionError, OverflowError):
        raise ValueError("Invalid tensor timing evidence") from None
