import json

import pytest
from test_build_sandbox import profile
from test_tensor_benchmark_timing import domain

from cannery.artifacts import ArtifactError, ArtifactRef, digest
from cannery.benchmark.contract import Cohort, cache_key, reference_key
from cannery.benchmark.tensor_evidence import read_tensor_timings, tensor_timing_artifact
from cannery.benchmark.tensor_timing import TENSOR_PROTOCOL


def values(tmp_path):
    worker = profile(tmp_path)
    cohort = Cohort(
        hardware_id="device4",
        driver="driver1",
        image=worker.image,
        loader_sha256="b" * 64,
        runtime=worker.fingerprint,
        protocol=TENSOR_PROTOCOL,
    )
    suite, contract = domain()
    samples = {"small": [1.5] * 100}
    raw = tensor_timing_artifact("a" * 64, suite, contract, cohort, samples)
    ref = ArtifactRef(
        repository="private", revision=digest(raw), sha256=digest(raw), path="timing.json"
    )
    return suite, contract, cohort, samples, raw, ref


def test_common_cohort_roundtrip_contains_only_timings(tmp_path):
    suite, contract, cohort, samples, raw, ref = values(tmp_path)
    assert read_tensor_timings(raw, ref, "a" * 64, suite, contract, cohort) == samples
    assert set(json.loads(raw)) == {"key", "timing"}
    assert json.loads(raw)["key"] == cache_key("a" * 64, suite, cohort)
    assert b"outputs" not in raw and b"inputs" not in raw
    for change in [{"hardware_id": "other"}, {"loader_sha256": "c" * 64}, {"driver": "other"}]:
        with pytest.raises(ValueError):
            read_tensor_timings(
                raw, ref, "a" * 64, suite, contract, cohort.model_copy(update=change)
            )
    with pytest.raises(ValueError):
        read_tensor_timings(raw, ref, "d" * 64, suite, contract, cohort)


def test_protocol_and_generator_cannot_cross_cohorts(tmp_path):
    suite, contract, cohort, samples, _, _ = values(tmp_path)
    unary = cohort.model_copy(update={"protocol": "sync-call-v1"})
    for generator, current in [("sha256-tensor-v2", unary), ("sha256-f32-v1", cohort)]:
        with pytest.raises(ValueError):
            reference_key("a" * 64, suite.sha256, generator, current)
    with pytest.raises(ValueError):
        tensor_timing_artifact("a" * 64, suite, contract, unary, samples)
    with pytest.raises(ValueError):
        Cohort.model_validate(cohort.model_dump() | {"device": "a3"})


@pytest.mark.parametrize("fault", ["missing", "extra", "bad_samples", "corrupt", "duplicate"])
def test_invalid_or_partial_timing_evidence_rejected(tmp_path, fault):
    suite, contract, cohort, samples, raw, ref = values(tmp_path)
    if fault in ("missing", "extra", "bad_samples"):
        if fault == "missing":
            samples = {}
        elif fault == "extra":
            samples["other"] = [1.0] * 100
        else:
            samples["small"][0] = False
        with pytest.raises(ValueError):
            tensor_timing_artifact("a" * 64, suite, contract, cohort, samples)
        return
    if fault == "corrupt":
        raw += b"corruption"
    else:
        raw = raw.replace(b'"key":', b'"key":null,"key":', 1)
        ref = ref.model_copy(update={"sha256": digest(raw), "revision": digest(raw)})
    with pytest.raises(ArtifactError if fault == "corrupt" else ValueError):
        read_tensor_timings(raw, ref, "a" * 64, suite, contract, cohort)
