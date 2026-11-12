"""Output-free private raw timing evidence with exact cohort and suite identity."""

import json

from cannery.artifacts import ArtifactRef, verified
from cannery.benchmark.contract import PROTOCOL, WARMUP, Cohort, cache_key, parse_samples
from cannery.numerics.recipes import RecipeSuite
from cannery.registry.manifest import unique_pairs


def timing_artifact(
    spec_sha256: str, suite: RecipeSuite, cohort: Cohort, samples: dict[str, list[float]]
) -> bytes:
    timing = {
        "protocol": PROTOCOL,
        "warmup": WARMUP,
        "cases": [
            {"id": c.id, "samples_us": samples[c.id]} for c in suite.cases if c.id in samples
        ],
    }
    checked = parse_samples(json.dumps(timing, allow_nan=False).encode(), suite)
    if set(checked) != set(samples):
        raise ValueError("Unexpected timing cases")
    return json.dumps(
        {"key": cache_key(spec_sha256, suite, cohort), "timing": timing},
        sort_keys=True,
        allow_nan=False,
    ).encode()


def read_timings(
    raw: bytes, artifact: ArtifactRef, spec_sha256: str, suite: RecipeSuite, cohort: Cohort
) -> dict[str, list[float]]:
    verified(raw, artifact)
    if len(raw) > 256 * 1024:
        raise ValueError("Timing evidence exceeds budget")
    value = json.loads(raw, object_pairs_hook=unique_pairs)
    if set(value) != {"key", "timing"} or value["key"] != cache_key(spec_sha256, suite, cohort):
        raise ValueError("Timing evidence identity differs")
    return parse_samples(json.dumps(value["timing"]).encode(), suite)
