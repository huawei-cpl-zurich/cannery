import base64
import hashlib
import json
import os
import struct
import subprocess
from uuid import uuid4

import pytest

from cannery.numerics.oracle_job import CPU_FLAGS
from cannery.numerics.oracle_worker import worker_source
from cannery.numerics.oracle_worker_v2 import worker_source_v2
from cannery.numerics.recipes_v2 import RecipeSuiteV2, _float_bytes, generate_v2
from cannery.numerics.tensor_contract import TensorContractV2


def test_old_worker_identity_and_new_supervisor_is_stdlib_only():
    assert hashlib.sha256(worker_source().encode()).hexdigest() == (
        "425585579d5864b093bf0ce83465a3a2090138571e6ca6dfe19830544ff8b412"
    )
    assert hashlib.sha256(worker_source(checkpoint=True).encode()).hexdigest() == (
        "386c3485506f68cc7a585ce414851c6a85d38e3c3ae85a2d9878eb99d7b3e382"
    )
    source = worker_source_v2()
    compile(source, "<tensor-oracle>", "exec")
    assert "import cannery" not in source and "import pydantic" not in source
    assert "def _generate_v2(" in source and "os.setuid(10001)" in source
    assert "readline(48*1024*1024)" in source


REFERENCE = """import os,torch
def reference(x,cache,indices,mask,enabled,offset):
    assert os.geteuid()==10001
    cache.add_(x.to(torch.float16) if enabled else 0)
    result={'state':cache,'sums':x.sum(dim=1),'indices':indices+offset,
            'bf16':x.to(torch.bfloat16),'flags':mask.clone()}
    # INJECT
    return result
"""


def plan(reference):
    cases, contracts = [], {}
    for identifier, rows, seed in [("small", 2, 1), ("empty", 0, 2), ("second_seed", 2, 3)]:
        inputs = {
            name: {
                "shape": shape,
                "dtype": dtype,
                "distribution": distribution,
                "seed": seed,
                **bounds,
            }
            for name, shape, dtype, distribution, bounds in [
                ("x", [rows, 3], "float32", "uniform", {}),
                ("cache", [rows, 3], "float16", "zeros", {}),
                ("indices", [rows], "int64", "randint", {"low": 0, "high": 7}),
                ("mask", [rows, 3], "bool", "randint", {"low": 0, "high": 1}),
            ]
        }
        cases.append({"id": identifier, "inputs": inputs})
        contracts[identifier] = {
            "outputs": {
                name: {"shape": shape, "dtype": dtype}
                for name, shape, dtype in [
                    ("state", [rows, 3], "float16"),
                    ("sums", [rows], "float32"),
                    ("indices", [rows], "int64"),
                    ("bf16", [rows, 3], "bfloat16"),
                    ("flags", [rows, 3], "bool"),
                ]
            },
            "scalars": {
                "enabled": {"kind": "bool", "value": True},
                "offset": {"kind": "int64", "value": 1},
            },
            "aliases": {"state": "cache"},
        }
    suite = RecipeSuiteV2.model_validate(
        {"schema_version": 2, "generator": "sha256-tensor-v2", "cases": cases}
    )
    contract = TensorContractV2.model_validate({"schema_version": 2, "cases": contracts})
    contract.validate_suite(suite)
    return suite, {
        "files": {"spec/reference.py": base64.b64encode(reference.encode()).decode()},
        "path": "spec",
        "environment": {"python": "3.12", "torch": "2.6.0+cpu"},
        "spec": {
            "reference": {"file": "reference.py", "function": "reference"},
            "validation": {"equal_nan": False},
        },
        "suite": suite.model_dump(mode="json"),
        "contract": contract.model_dump(mode="json"),
    }


@pytest.mark.parametrize(
    "fault", [None, "raises", "mutates", "alias", "rebind", "dtype", "shape", "nan", "hang"]
)
def test_real_isolated_multi_output_reference(tmp_path, fault):
    image = os.environ.get("CANNERY_ORACLE_TEST_IMAGE")
    if not image:
        pytest.skip("Set CANNERY_ORACLE_TEST_IMAGE for isolated tensor CPU oracle evidence")
    injection = {
        None: "pass",
        "raises": "raise RuntimeError('injected')",
        "mutates": "x.add_(1)",
        "alias": "result['state']=cache.clone()",
        "rebind": "cache.set_(cache.clone())",
        "dtype": "result['indices']=indices.float()",
        "shape": "result['bf16']=result['bf16'].flatten()",
        "nan": "result['bf16'].fill_(float('nan'))",
        "hang": "while True: pass",
    }[fault]
    suite, request = plan(REFERENCE.replace("# INJECT", injection))
    tmp_path.chmod(0o755)
    (tmp_path / "plan.json").write_text(json.dumps(request))
    source = worker_source_v2()
    if fault == "hang":
        source = source.replace("signal.alarm(70)", "signal.alarm(10)")
    name = "cannery-llm-catalog-oracle-" + uuid4().hex
    arguments = [
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
    ]
    try:
        result = subprocess.run(arguments, capture_output=True, timeout=85)
        if fault:
            assert result.returncode != 0
            assert b'"outputs"' not in result.stdout
            if fault == "hang":
                assert result.returncode == 124
            return
        assert result.returncode == 0, result.stderr.decode(errors="replace")
        outputs = json.loads(result.stdout)["outputs"]
        for case in suite.cases:
            values = {
                name: base64.b64decode(data, validate=True)
                for name, data in outputs[case.id].items()
            }
            x = generate_v2(case.inputs["x"], case.id, "x")
            floats = [v for (v,) in struct.iter_unpack("<f", x)]
            assert values["state"] == b"".join(struct.pack("<e", v) for v in floats)
            assert values["bf16"] == b"".join(_float_bytes(v, "bfloat16") for v in floats)
            assert values["flags"] == generate_v2(case.inputs["mask"], case.id, "mask")
            indices = generate_v2(case.inputs["indices"], case.id, "indices")
            assert values["indices"] == b"".join(
                struct.pack("<q", v + 1) for (v,) in struct.iter_unpack("<q", indices)
            )
            expected = [sum(floats[i : i + 3]) for i in range(0, len(floats), 3)]
            actual = [v for (v,) in struct.iter_unpack("<f", values["sums"])]
            assert actual == pytest.approx(expected, abs=1e-6)
        assert set(outputs) == {case.id for case in suite.cases}
        assert {p.name for p in tmp_path.iterdir()} == {"plan.json"}
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=15)
