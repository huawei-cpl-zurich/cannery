import copy
import json

import pytest
from test_tensor_numerical_contract import fixture

from cannery.benchmark.contract import SAMPLES, WARMUP
from cannery.benchmark.tensor_timing import (
    TENSOR_PROTOCOL,
    measure_tensor,
    parse_tensor_samples,
    tensor_timed_cases,
)
from cannery.numerics.recipes_v2 import RecipeSuiteV2
from cannery.numerics.tensor_contract import TensorContractV2


def test_mutation_reset_and_transfers_are_outside_every_timed_interval():
    current, now, clocks, observed = [0], [0], [], []

    def reset():
        current[0] = 7
        now[0] += 500_000

    def synchronize():
        now[0] += 1000

    def forward():
        assert current[0] == 7
        current[0] += 3
        observed.append(current[0])
        now[0] += 2000
        return current

    def snapshot(value):
        now[0] += 200_000
        return list(value)

    def clock():
        clocks.append(now[0])
        return now[0]

    samples, before, after = measure_tensor(forward, reset, synchronize, clock, snapshot=snapshot)
    assert samples == [3.0] * SAMPLES
    assert before == after == [10]
    current[0] = 0
    assert before == after == [10]
    assert len(observed) == WARMUP + SAMPLES + 2 and len(clocks) == 2 * SAMPLES


@pytest.mark.parametrize("stage", ["reset", "synchronize", "forward", "snapshot"])
def test_device_or_reset_failure_propagates_without_retrying_or_returning_samples(stage):
    calls = {name: 0 for name in ["reset", "synchronize", "forward", "snapshot"]}

    def operation(name):
        def run(*args):
            calls[name] += 1
            if name == stage:
                raise RuntimeError("injected device error")
            return object()

        return run

    with pytest.raises(RuntimeError, match="injected device error"):
        measure_tensor(
            operation("forward"),
            operation("reset"),
            operation("synchronize"),
            lambda: 0,
            snapshot=operation("snapshot"),
        )
    assert calls[stage] == 1


def domain():
    suite, contract = fixture()
    return RecipeSuiteV2.model_validate(suite), TensorContractV2.model_validate(contract)


def payload():
    return {
        "protocol": TENSOR_PROTOCOL,
        "warmup": WARMUP,
        "cases": [{"id": "small", "samples_us": [1.25] * SAMPLES}],
    }


def test_complete_protocol_roundtrip_and_entirely_empty_cases():
    suite, contract = domain()
    assert parse_tensor_samples(json.dumps(payload()).encode(), suite, contract) == {
        "small": [1.25] * SAMPLES
    }
    raw_suite, raw_contract = fixture()
    for desc in raw_suite["cases"][0]["inputs"].values():
        desc["shape"][0] = 0
    for desc in raw_contract["cases"]["small"]["outputs"].values():
        desc["shape"][0] = 0
    empty_suite = RecipeSuiteV2.model_validate(raw_suite)
    empty_contract = TensorContractV2.model_validate(raw_contract)
    assert tensor_timed_cases(empty_suite, empty_contract) == []
    # A nonempty secondary input still makes the call a timed case.
    raw_suite["cases"][0]["inputs"]["cache"]["shape"][0] = 4
    raw_contract["cases"]["small"]["outputs"]["updated"]["shape"][0] = 4
    assert tensor_timed_cases(
        RecipeSuiteV2.model_validate(raw_suite), TensorContractV2.model_validate(raw_contract)
    ) == ["small"]


@pytest.mark.parametrize(
    "change", ["protocol", "warmup", "missing", "extra", "order", "bool", "nan", "zero", "count"]
)
def test_invalid_or_partial_samples_never_cross_controller_boundary(change):
    suite, contract = domain()
    value = copy.deepcopy(payload())
    if change == "protocol":
        value["protocol"] = "sync-call-v1"
    elif change == "warmup":
        value["warmup"] = True
    elif change == "missing":
        value["cases"] = []
    elif change == "extra":
        value["cases"][0]["unexpected"] = 0
    elif change == "order":
        value["cases"][0]["id"] = "other"
    elif change == "bool":
        value["cases"][0]["samples_us"][0] = True
    elif change == "nan":
        value["cases"][0]["samples_us"][0] = float("nan")
    elif change == "zero":
        value["cases"][0]["samples_us"][0] = 0
    else:
        value["cases"][0]["samples_us"].pop()
    with pytest.raises(ValueError):
        parse_tensor_samples(json.dumps(value).encode(), suite, contract)


def test_duplicate_json_fields_and_oversized_messages_rejected():
    suite, contract = domain()
    for raw in [b"x" * (256 * 1024 + 1), b'{"protocol":"x","protocol":"x"}']:
        with pytest.raises(ValueError):
            parse_tensor_samples(raw, suite, contract)
