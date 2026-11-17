import copy
import json

import pytest
from test_release_contract import release_data  # noqa: F401

from cannery.artifacts import digest
from cannery.benchmark.contract import Cohort, reference_key
from cannery.releases.tensor_contract import TensorReleaseManifest, read_tensor_manifest


@pytest.fixture
def tensor_release(release_data):  # noqa: F811
    data = release_data
    data.update(schema_version=2, generator="sha256-tensor-v2", abi="tensor-v2", language="triton")
    data["cohort"]["protocol"] = data["score"]["protocol"] = "sync-call-tensor-v2"
    data["interface"] = {"inputs": ["x", "cache"], "outputs": ["updated", "indices"]}
    data["reference_key"] = reference_key(
        data["spec_sha256"], data["suite_sha256"], data["generator"], Cohort(**data["cohort"])
    )
    data["cases"] = {}
    for row in data["score"]["cases"]:
        shape = row["shape"]
        tensor = {"shape": shape, "dtype": "float16"}
        data["cases"][row["id"]] = {
            "inputs": {"x": copy.deepcopy(tensor), "cache": copy.deepcopy(tensor)},
            "outputs": {
                "updated": copy.deepcopy(tensor),
                "indices": {"shape": shape, "dtype": "int32"},
            },
            "aliases": {"updated": "cache"},
            "scalars": {"offset": {"kind": "int64", "value": 1}},
        }
    return data


def parse(data):
    return TensorReleaseManifest.model_validate_json(json.dumps(data))


@pytest.mark.parametrize("language", ["ascendc", "catlass", "triton"])
def test_named_mixed_mutable_release_roundtrip_without_outputs(tensor_release, language):
    tensor_release["language"] = language
    manifest = parse(tensor_release)
    raw = manifest.encoded()
    assert read_tensor_manifest(raw, digest(raw)) == manifest
    assert manifest.cases["small"].aliases == {"updated": "cache"}
    assert manifest.cases["small"].outputs["indices"].dtype == "int32"
    assert manifest.score.geomean_speedup == 1
    assert manifest.cohort_sha256 == parse(copy.deepcopy(tensor_release)).cohort_sha256


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(abi="unary-f32-v1"),
        lambda d: d.update(generator="sha256-f32-v1"),
        lambda d: d.update(device="a3"),
        lambda d: d.update(reference_key="0" * 64),
        lambda d: d["package"].update(revision="0" * 64),
        lambda d: d["cohort"].update(protocol="sync-call-v1"),
        lambda d: d["interface"].update(inputs=["cache", "missing"]),
        lambda d: d["cases"]["small"]["aliases"].update(updated="missing"),
        lambda d: d["cases"]["small"]["outputs"]["updated"].update(dtype="float32"),
        lambda d: d["cases"]["small"]["scalars"]["offset"].update(value=True),
        lambda d: d["cases"]["small"]["outputs"]["indices"].update(values=[1, 2]),
        lambda d: d["score"]["cases"][0].update(shape=[3, 2]),
        lambda d: d["score"]["cases"][0].update(excluded="empty_correctness_only"),
        lambda d: d["score"].update(geomean_speedup=2),
        lambda d: d["score"]["cases"].pop(),
        lambda d: d["tolerances"].append(copy.deepcopy(d["tolerances"][0])),
    ],
)
def test_inconsistent_or_output_bearing_tensor_release_rejected(tensor_release, mutate):
    mutate(tensor_release)
    with pytest.raises(ValueError):
        parse(tensor_release)


def test_empty_first_input_with_nonempty_output_remains_timed(tensor_release):
    d = tensor_release
    d["cases"]["small"]["inputs"]["x"]["shape"] = [0, 3]
    d["score"]["cases"][0]["shape"] = [0, 3]
    assert parse(d).score.cases[0].speedup == 1
    d["score"]["cases"][0] = dict(id="small", shape=[0, 3], excluded="empty_correctness_only")
    with pytest.raises(ValueError):
        parse(d)


def test_slow_results_canonical_digest_and_bounded_reader(tensor_release):
    row = tensor_release["score"]["cases"][0]
    row["candidate"] = dict(median_us=20.0, p95_us=22.0, samples=100)
    row["speedup"] = tensor_release["score"]["geomean_speedup"] = 0.5
    manifest = parse(tensor_release)
    raw = manifest.encoded()
    assert read_tensor_manifest(raw, digest(raw)).score.geomean_speedup == 0.5
    for bad, pin in [
        (raw, "0" * 64),
        (raw + b" ", digest(raw + b" ")),
        (
            b'{"schema_version":2,"schema_version":2}',
            digest(b'{"schema_version":2,"schema_version":2}'),
        ),
        (b"x" * 262145, digest(b"x" * 262145)),
    ]:
        with pytest.raises(ValueError):
            read_tensor_manifest(bad, pin)
