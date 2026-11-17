import base64
import hashlib
import json
import os
import subprocess
from uuid import uuid4

import pytest
from test_tensor_oracle_worker import REFERENCE, plan

from cannery.numerics.oracle_job import CPU_FLAGS
from cannery.numerics.recipes_v2 import generate_v2, input_sha256_v2
from cannery.numerics.validation_runtime import runtime_program
from cannery.numerics.validation_runtime_v2 import TENSOR_RUNTIME, tensor_runtime_program


def test_standalone_runtime_preserves_generator_and_legacy_program():
    source = tensor_runtime_program()
    compile(source, "<tensor-runtime>", "exec")
    assert "import cannery." not in source and "import pydantic" not in source
    assert "baseline.tar" not in source and "expected_outputs" not in source
    assert source.index("wire=os.fdopen") < source.index("import torch,torch_npu")
    prefix = source.split("# Reserve the result channel")[0]
    namespace = {}
    exec(prefix, namespace)
    suite, _ = plan(REFERENCE)
    for case in suite.cases:
        for name, recipe in case.inputs.items():
            assert namespace["_generate_v2"](recipe, case.id, name) == generate_v2(
                recipe, case.id, name
            )
            data = generate_v2(recipe, case.id, name)
            assert namespace["input_sha256_v2"](recipe, data) == input_sha256_v2(recipe, data)
    assert hashlib.sha256(runtime_program().encode()).hexdigest() == (
        "88933cc81759435ba90a365ef48999fc6ac436b9db344a2e3b7aa31ad89a9b62"
    )


@pytest.mark.parametrize(
    "fault",
    [None, "readonly", "rebind", "alias", "shape", "launch", "sync", "format", "format_cast"],
)
def test_cpu_device_simulation_and_faults(tmp_path, fault):
    image = os.environ.get("CANNERY_ORACLE_TEST_IMAGE")
    if not image:
        pytest.skip("Set CANNERY_ORACLE_TEST_IMAGE for isolated CPU device simulation")
    suite, request = plan(REFERENCE)
    tmp_path.chmod(0o755)
    request["input_sha256"] = {
        case.id: {
            name: input_sha256_v2(recipe, generate_v2(recipe, case.id, name))
            for name, recipe in case.inputs.items()
        }
        for case in suite.cases
    }
    (tmp_path / "cases.json").write_text(json.dumps(request))
    injection = {
        None: "pass",
        "readonly": "x.add_(1)",
        "rebind": "cache.set_(cache.clone())",
        "alias": "result['state']=cache.clone()",
        "shape": "result['bf16']=result['bf16'].flatten()",
        "launch": "raise RuntimeError('injected launch failure')",
        "sync": "pass",
        "format": "pass",
        "format_cast": "pass",
    }[fault]
    reference = REFERENCE.replace("    assert os.geteuid()==10001\n", "").replace(
        "# INJECT", injection
    )
    source = (
        tensor_runtime_program().split("# Reserve the result channel")[0]
        + """
import torch
wire=sys.stdout.buffer
# CPU simulation only: this is not hardware evidence.
formats={}
def allocate(shape,dtype,**kwargs):
    value=torch.empty(shape,dtype=dtype)
    formats[id(value)]=INITIAL_FORMAT
    return value
def normalize(value,format):
    if CAST_FAILURE:raise RuntimeError('injected format conversion device failure')
    formats[id(value)]=format
    return value
torch_npu=SimpleNamespace(get_npu_format=lambda value:formats.get(id(value),2),
    empty_with_format=allocate,npu_format_cast=normalize)
def synchronize():
    SYNC
""".replace(
            "    SYNC",
            "    raise RuntimeError('injected device loss')" if fault == "sync" else "    pass",
        )
    )
    source = source.replace("INITIAL_FORMAT", "0" if fault in ("format", "format_cast") else "2")
    source = source.replace("CAST_FAILURE", repr(fault == "format_cast"))
    source += "\ntorch.npu=SimpleNamespace(synchronize=synchronize)\n" + reference
    source += """
class Kernel:
    metadata={'schema_version':3,'abi':'tensor-v2', 'variants':[{'inputs':[
        {'name':name} for name in ['x','cache','indices','mask']]}]}
    forward=staticmethod(reference)
kernel=Kernel()
"""
    source += TENSOR_RUNTIME
    name = "cannery-llm-catalog-device-sim-" + uuid4().hex
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
        if fault not in (None, "readonly", "format"):
            assert result.returncode != 0 and not result.stdout
            assert (
                b"injected" if fault in ("launch", "sync", "format_cast") else b"AssertionError"
            ) in result.stderr
            return
        assert result.returncode == 0, result.stderr.decode(errors="replace")
        reply = json.loads(result.stdout)
        assert reply["schema_version"] == 2 and len(reply["cases"]) == 3
        for case, output in zip(suite.cases, reply["cases"], strict=True):
            assert output["id"] == case.id
            for input_name, expected in request["input_sha256"][case.id].items():
                unchanged = output["inputs"][input_name] == expected
                if case.id == "empty" or input_name not in ("cache", "x"):
                    assert unchanged
                elif input_name == "x":
                    assert unchanged == (fault != "readonly")
            assert base64.b64decode(output["outputs"]["flags"]["data"]) == generate_v2(
                case.inputs["mask"], case.id, "mask"
            )
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=15)


def test_tensor_bootstrap_bounds_cpu_threads_before_framework_import():
    image = os.environ.get("CANNERY_ORACLE_TEST_IMAGE")
    if not image:
        pytest.skip("Set CANNERY_ORACLE_TEST_IMAGE for actual CPU thread-budget check")
    source = tensor_runtime_program().split("# Reserve the result channel", 1)[0]
    source += "import torch\nassert torch.get_num_threads()==1\nprint('THREAD_BUDGET_PASS')\n"
    name = "cannery-llm-catalog-thread-budget-" + uuid4().hex
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
                image,
                "python",
                "-I",
                "-c",
                source,
            ],
            capture_output=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr.decode(errors="replace")
        assert result.stdout == b"THREAD_BUDGET_PASS\n"
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=15)
