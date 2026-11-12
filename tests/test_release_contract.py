import copy
import json
from uuid import UUID

import pytest

from cannery.artifacts import digest
from cannery.releases.contract import ReleaseManifest, read_manifest


@pytest.fixture
def release_data():
    artifact = dict(repository="private", revision="a" * 64, path="binary.tar", sha256="a" * 64)
    latency = dict(median_us=10.0, p95_us=12.0, samples=100)
    return dict(
        submission_id=str(UUID(int=1)),
        kernel="team/example",
        spec_version=1,
        spec_sha256="b" * 64,
        suite_sha256="c" * 64,
        generator="sha256-f32-v1",
        source=dict(url="https://example.org/team/repo.git", commit="1" * 40, path="kernel"),
        source_sha256="d" * 64,
        language="ascendc",
        abi="unary-f32-v1",
        package=artifact,
        numerical_attempt=str(UUID(int=2)),
        benchmark_attempt=str(UUID(int=3)),
        reference_attempt=str(UUID(int=4)),
        reference_key="e" * 64,
        timing_artifact=artifact,
        reference_artifact=artifact,
        cohort=dict(
            hardware_id="device",
            driver="driver",
            image="sha256:" + "a" * 64,
            loader_sha256="f" * 64,
            runtime=dict(
                machine="aarch64",
                python="3.12",
                torch="2.9.0",
                torch_npu="2.9.0",
                cann="9.3.0",
                soc="Ascend950",
            ),
        ),
        tolerances=[dict(rtol=1e-5, atol=1e-6)],
        score=dict(
            protocol="sync-call-v1",
            cases=[
                dict(id="small", shape=[2, 3], reference=latency, candidate=latency, speedup=1.0),
                dict(id="empty", shape=[0, 3], excluded="empty_correctness_only"),
            ],
            geomean_speedup=1.0,
        ),
    )


def parse(data):
    return ReleaseManifest.model_validate_json(json.dumps(data))


def test_pinned_canonical_roundtrip_and_corruption(release_data):
    manifest = parse(release_data)
    raw = manifest.encoded()
    assert read_manifest(raw, digest(raw)) == manifest
    assert manifest.cohort_sha256 == parse(dict(reversed(list(release_data.items())))).cohort_sha256
    for bad, pin in [
        (raw + b" ", digest(raw)),
        (b"x" * 131073, digest(b"x" * 131073)),
        (
            b'{"schema_version":1,"schema_version":1}',
            digest(b'{"schema_version":1,"schema_version":1}'),
        ),
    ]:
        with pytest.raises(ValueError):
            read_manifest(bad, pin)
    indented = json.dumps(json.loads(raw), indent=2).encode()
    with pytest.raises(ValueError):
        read_manifest(indented, digest(indented))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["package"].update(revision="b" * 64),
        lambda d: d.update(language="catlass"),
        lambda d: d["score"].update(geomean_speedup=2.0),
        lambda d: d["score"]["cases"][0].update(speedup=0.0),
        lambda d: d["score"]["cases"][0]["candidate"].update(samples=99),
        lambda d: d["score"]["cases"][0]["candidate"].update(p95_us=1.0),
        lambda d: d["score"]["cases"][1].update(speedup=1.0),
        lambda d: d["score"]["cases"].append(copy.deepcopy(d["score"]["cases"][0])),
        lambda d: d["tolerances"].append(copy.deepcopy(d["tolerances"][0])),
        lambda d: d["tolerances"][0].update(atol=float("nan")),
        lambda d: d["cohort"]["runtime"].pop("cann"),
        lambda d: d.update(outputs={"small": [1, 2, 3]}),
    ],
)
def test_invalid_release_never_loads(release_data, mutate):
    mutate(release_data)
    with pytest.raises(ValueError):
        parse(release_data)


def test_slow_and_scoreless_releases_remain_representable(release_data):
    d = release_data
    d["score"]["cases"][0]["candidate"] = dict(median_us=20.0, p95_us=22.0, samples=100)
    d["score"]["cases"][0]["speedup"] = d["score"]["geomean_speedup"] = 0.5
    assert parse(d).score.geomean_speedup == 0.5
    d["score"]["cases"] = [d["score"]["cases"][1]]
    d["score"]["geomean_speedup"] = None
    assert parse(d).score.geomean_speedup is None


def test_runtime_and_static_specialization_coverage(release_data):
    d = release_data
    runtime = d["cohort"]["runtime"]
    assert parse(d).supports(runtime, (99,))
    assert not parse(d).supports(runtime | {"cann": "other"}, (2, 3))
    d.update(
        language="catlass",
        abi="catlass-unary-f32-v1",
        specializations=[dict(shape=[2, 3], dtype="float32"), dict(shape=[0, 3], dtype="float32")],
    )
    assert parse(d).supports(runtime, (2, 3))
    assert not parse(d).supports(runtime, (3, 2))
    d["specializations"].pop()
    with pytest.raises(ValueError):
        parse(d)
