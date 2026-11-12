import copy
import json

import pytest
from test_validation_runtime import protocol

from cannery.benchmark.contract import PROTOCOL
from cannery.benchmark.runtime import parse_runtime, runtime_program


def envelope():
    suite, tensors = protocol()
    payload = {
        "protocol": PROTOCOL,
        "timing": {
            "protocol": PROTOCOL,
            "warmup": 20,
            "cases": [{"id": c.id, "samples_us": [1.0] * 100} for c in suite.cases[:-1]],
        },
        "before": tensors,
        "after": copy.deepcopy(tensors),
    }
    return suite, payload


@pytest.mark.parametrize("kind", ["reference", "candidate"])
def test_generated_runner_is_self_contained_and_offline(kind):
    source = runtime_program(kind)
    compile(source, "<benchmark>", "exec")
    assert "import cannery." not in source
    assert "torch.no_grad()" in source and "time.perf_counter_ns" in source
    assert "backing=torch.empty" in source and "x.stride()==strides" in source
    assert "expected_outputs" not in source and "baseline.tar" not in source
    if kind == "reference":
        assert "reference(**" in source and "admitted.tar" not in source
    else:
        assert "load_cannery_package" in source and "module_spec.loader" not in source


def test_before_after_and_timing_are_strictly_bound_to_suite():
    suite, payload = envelope()
    result = parse_runtime(json.dumps(payload).encode(), suite)
    assert set(result.samples) == {"small", "strided"}
    assert result.before == result.after and result.before["empty"].data == b""


@pytest.mark.parametrize("fault", ["missing", "extra", "protocol", "timing", "tensor", "duplicate"])
def test_malformed_result_never_scores(fault):
    suite, p = envelope()
    match fault:
        case "missing":
            del p["before"]
        case "extra":
            p["passed"] = True
        case "protocol":
            p["protocol"] = "other"
        case "timing":
            p["timing"]["cases"][0]["samples_us"][0] = 0
        case "tensor":
            p["after"]["cases"][0]["data"] = "AA=="
        case "duplicate":
            raw = json.dumps(p).replace('"protocol":', '"protocol":"wrong","protocol":', 1).encode()
    if fault != "duplicate":
        raw = json.dumps(p).encode()
    with pytest.raises(ValueError, match="Invalid benchmark output"):
        parse_runtime(raw, suite)


def test_invalid_runner_kind_fails_early():
    with pytest.raises(ValueError):
        runtime_program("unknown")
