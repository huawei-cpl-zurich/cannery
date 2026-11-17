import copy
import hashlib
import json
import os
import subprocess
from uuid import uuid4

import pytest
from test_tensor_oracle_worker import REFERENCE, plan
from test_tensor_validation_protocol import protocol

from cannery.benchmark.contract import SAMPLES, WARMUP
from cannery.benchmark.runtime import runtime_program
from cannery.benchmark.tensor_runtime import RUN, parse_tensor_runtime, tensor_benchmark_program
from cannery.benchmark.tensor_timing import TENSOR_PROTOCOL
from cannery.numerics.oracle_job import CPU_FLAGS
from cannery.numerics.recipes_v2 import generate_v2, input_sha256_v2
from cannery.numerics.tensor_contract import TensorContractV2


def test_standalone_programs_compile_and_preserve_unary_program():
    for kind in ["candidate", "reference"]:
        source = tensor_benchmark_program(kind)
        compile(source, "<tensor-benchmark>", "exec")
        assert "import cannery." not in source and "import pydantic" not in source
        assert "expected_outputs" not in source and "baseline.tar" not in source
    with pytest.raises(ValueError):
        tensor_benchmark_program("wrong")
    # Existing unary program remains generated from its original code.
    assert hashlib.sha256(runtime_program("candidate").encode()).hexdigest() == (
        "4e31a2ed538de7bfd5ad2e50151566ea3c125448e334df563c426c356ad46824"
    )


def wire():
    suite, contract, values = protocol()
    return (
        suite,
        contract,
        {
            "protocol": TENSOR_PROTOCOL,
            "timing": {
                "protocol": TENSOR_PROTOCOL,
                "warmup": WARMUP,
                "cases": [{"id": case.id, "samples_us": [2.0] * SAMPLES} for case in suite.cases],
            },
            "before": values,
            "after": copy.deepcopy(values),
        },
    )


def test_timing_and_named_outputs_both_cross_trusted_parser():
    suite, contract, value = wire()
    result = parse_tensor_runtime(json.dumps(value).encode(), suite, contract)
    assert list(result.samples) == ["small", "second"]
    assert result.before.outputs == result.after.outputs
    assert result.before.input_hashes == result.after.input_hashes


@pytest.mark.parametrize("fault", ["protocol", "timing", "before", "after", "extra", "duplicate"])
def test_partial_or_invalid_tensor_timing_is_never_accepted(fault):
    suite, contract, value = wire()
    if fault == "protocol":
        value["protocol"] = "sync-call-v1"
    elif fault == "timing":
        value["timing"]["cases"].pop()
    elif fault in ("before", "after"):
        value[fault]["cases"].pop()
    elif fault == "extra":
        value["extra"] = 1
    raw = json.dumps(value).encode()
    if fault == "duplicate":
        raw = raw.replace(b'"before":', b'"before":null,"before":', 1)
    with pytest.raises(ValueError):
        parse_tensor_runtime(raw, suite, contract)


@pytest.mark.parametrize("fault", [None, "readonly", "alias", "shape", "timed_launch", "sync"])
def test_cpu_device_simulation_resets_each_call_and_retains_full_evidence(tmp_path, fault):
    image = os.environ.get("CANNERY_ORACLE_TEST_IMAGE")
    if not image:
        pytest.skip("Set CANNERY_ORACLE_TEST_IMAGE for isolated CPU device simulation")
    suite, request = plan(REFERENCE)
    request["input_order"] = ["x", "cache", "indices", "mask"]
    request["input_sha256"] = {
        case.id: {
            name: input_sha256_v2(recipe, generate_v2(recipe, case.id, name))
            for name, recipe in case.inputs.items()
        }
        for case in suite.cases
    }
    tmp_path.chmod(0o755)
    (tmp_path / "cases.json").write_text(json.dumps(request))
    injection = {
        None: "pass",
        "readonly": "x.add_(1)",
        "alias": "result['state']=cache.clone()",
        "shape": "result['bf16']=result['bf16'].flatten()",
        "timed_launch": (
            "\n    global calls\n    calls+=1\n"
            "    if calls==30: raise RuntimeError('injected timed launch failure')"
        ),
        "sync": "pass",
    }[fault]
    reference = REFERENCE.replace(
        "    assert os.geteuid()==10001",
        '    assert bool((cache==0).all()), "mutable input was not reset"',
    ).replace("# INJECT", injection)
    source = tensor_benchmark_program("candidate").split("wire=os.fdopen", 1)[0]
    source += """
import torch
wire=sys.stdout.buffer
plan=json.loads(Path('/input/cases.json').read_bytes())
# CPU device simulation only; no hardware timing claims.
torch_npu=SimpleNamespace(get_npu_format=lambda _:2,
    empty_with_format=lambda shape,dtype,**_:torch.empty(shape,dtype=dtype))
sync_calls=0
calls=0
def synchronize():
    global sync_calls
    sync_calls+=1
    SYNC
torch.npu=SimpleNamespace(synchronize=synchronize)
""".replace(
        "    SYNC",
        "    if sync_calls==30: raise RuntimeError('injected sync failure')"
        if fault == "sync"
        else "    pass",
    )
    source += "\n" + reference + "\nforward=reference\n"
    from cannery.numerics.validation_runtime_v2 import TENSOR_RUNTIME

    source += "def raw" + TENSOR_RUNTIME.split("def raw", 1)[1].split("\n\nresults=[]", 1)[0] + RUN
    name = "cannery-llm-catalog-timing-sim-" + uuid4().hex
    try:
        result = subprocess.run(
            [
                "docker",
                "run",
                *CPU_FLAGS[1:],
                "--name",
                name,
                "--label",
                "cannery.task=llm-catalog",
                "--mount",
                f"type=bind,src={tmp_path},dst=/input,readonly",
                image,
                "python",
                "-I",
                "-c",
                source,
            ],
            capture_output=True,
            timeout=30,
        )
        if fault not in (None, "readonly"):
            assert result.returncode != 0 and not result.stdout
            assert (
                b"injected" if fault in ("timed_launch", "sync") else b"AssertionError"
            ) in result.stderr
            return
        assert result.returncode == 0, result.stderr.decode(errors="replace")
        contract = TensorContractV2.model_validate(request["contract"])
        checked = parse_tensor_runtime(result.stdout, suite, contract)
        assert set(checked.samples) == {"small", "second_seed"}
        if fault is None:
            assert checked.before.outputs == checked.after.outputs
        for side in [checked.before, checked.after]:
            for case in suite.cases:
                unchanged = side.input_hashes[case.id]["x"] == request["input_sha256"][case.id]["x"]
                assert unchanged == (fault != "readonly" or case.id == "empty")
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=15)
