import json

import pytest
from test_recipe_spec import recipe_spec

from cannery.benchmark.contract import (
    PROTOCOL,
    Cohort,
    cache_key,
    measure,
    parse_samples,
    score,
    summarize,
)
from cannery.numerics.recipes import parse_recipes


@pytest.fixture
def suite():
    files, _, _ = recipe_spec()
    return parse_recipes(files["spec/recipes.yaml"])


def test_synchronized_call_measures_only_call_and_completion():
    events = []
    now = 0

    def clock():
        return now

    def forward(x):
        nonlocal now
        events.append("call")
        now += 3000
        return x + 1

    def sync():
        nonlocal now
        events.append("sync")
        now += 2000

    samples, before, after = measure(forward, 1, sync, clock)
    assert samples == [5.0] * 100 and before == after == 2
    assert events.count("call") == 122
    assert events[42:-2] == ["sync", "call", "sync"] * 100


def test_statistics_and_complete_suite_geomean(suite):
    a, b, empty = [c.id for c in suite.cases]
    result = score(suite, {a: [20.0] * 100, b: [5.0] * 100}, {a: [10.0] * 100, b: [10.0] * 100})
    assert result["geomean_speedup"] == 1.0
    assert result["cases"][2] == {
        "id": empty,
        "shape": [0, 3],
        "excluded": "empty_correctness_only",
    }
    assert summarize(list(range(1, 101))) == {"median_us": 50.5, "p95_us": 95, "samples": 100}
    with pytest.raises(ValueError):
        score(suite, {a: [1.0] * 100}, {a: [1.0] * 100})
    assert json.loads(json.dumps(result, allow_nan=False)) == result


@pytest.mark.parametrize("bad", [0, -1, True, float("nan"), float("inf"), "1", None, 10**400])
def test_invalid_samples_never_score(bad):
    with pytest.raises(ValueError):
        summarize([bad] * 100)


@pytest.mark.parametrize("count", [0, 99, 101])
def test_sample_count_is_exact(count):
    with pytest.raises(ValueError):
        summarize([1.0] * count)


def payload(suite):
    return {
        "protocol": PROTOCOL,
        "warmup": 20,
        "cases": [{"id": c.id, "samples_us": [1.0] * 100} for c in suite.cases[:-1]],
    }


def test_strict_wire_and_all_empty_suite(suite):
    wire = payload(suite)
    assert set(parse_samples(json.dumps(wire).encode(), suite)) == {c.id for c in suite.cases[:-1]}
    empty = suite.model_copy(update={"cases": (suite.cases[-1],)})
    assert score(empty, {}, {})["geomean_speedup"] is None
    assert (
        parse_samples(json.dumps({"protocol": PROTOCOL, "warmup": 20, "cases": []}).encode(), empty)
        == {}
    )


@pytest.mark.parametrize(
    "fault", ["duplicate", "extra", "missing", "order", "protocol", "warmup", "unknown"]
)
def test_wire_rejects_incomplete_or_ambiguous_evidence(suite, fault):
    p = payload(suite)
    match fault:
        case "duplicate":
            raw = json.dumps(p).replace('"warmup": 20', '"warmup": 20, "warmup": 20').encode()
        case "extra":
            p["cases"].append({"id": "extra", "samples_us": [1.0] * 100})
        case "missing":
            p["cases"].pop()
        case "order":
            p["cases"].reverse()
        case "protocol":
            p["protocol"] = "other"
        case "warmup":
            p["warmup"] = True
        case "unknown":
            p["cases"][0]["passed"] = True
    if fault != "duplicate":
        raw = json.dumps(p).encode()
    with pytest.raises(ValueError):
        parse_samples(raw, suite)


def test_cohort_key_binds_all_measurement_identity(suite):
    cohort = Cohort(
        hardware_id="serial:chip2",
        driver="driver1",
        image="sha256:" + "a" * 64,
        loader_sha256="b" * 64,
        runtime={
            "machine": "aarch64",
            "python": "3.11",
            "torch": "2.6",
            "torch_npu": "2.6",
            "cann": "9.3",
            "soc": "Ascend950",
        },
    )
    original = cache_key("c" * 64, suite, cohort)
    for field, value in [
        ("hardware_id", "other"),
        ("driver", "driver2"),
        ("image", "sha256:" + "d" * 64),
        ("loader_sha256", "e" * 64),
        ("runtime", {"torch": "2.7", "cann": "9.3"}),
    ]:
        assert cache_key("c" * 64, suite, cohort.model_copy(update={field: value})) != original
    assert cache_key("d" * 64, suite, cohort) != original
    with pytest.raises(ValueError):
        cache_key("bad", suite, cohort)


@pytest.mark.parametrize("sample", [5e-324, 1e308])
def test_extreme_finite_samples_keep_positive_finite_median(suite, sample):
    timings = {c.id: [sample] * 100 for c in suite.cases[:-1]}
    assert summarize([sample] * 100)["median_us"] == sample
    assert score(suite, timings, timings)["geomean_speedup"] == 1.0
