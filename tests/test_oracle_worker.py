import base64
import json
import os
import struct
import subprocess
from uuid import uuid4

import pytest
from test_recipe_spec import recipe_spec

from cannery.numerics.oracle_worker import worker_source
from cannery.numerics.recipes import generate, parse_recipes
from cannery.registry.manifest import validate_spec


def test_generated_supervisor_has_no_control_plane_or_author_import():
    source = worker_source()
    compile(source, "<trusted-oracle>", "exec")
    assert "import cannery" not in source and "import pydantic" not in source
    assert "def _generate(" in source and "os.setuid(10001)" in source


@pytest.mark.parametrize("fault", [None, "raises", "mutates", "wrong_dtype", "nan", "negative"])
def test_real_cpu_oracle_isolates_author_and_returns_outputs_only(tmp_path, fault):
    image = os.environ.get("CANNERY_ORACLE_TEST_IMAGE")
    if not image:
        pytest.skip("Set CANNERY_ORACLE_TEST_IMAGE to run isolated Docker CPU evidence")
    files, _, _ = recipe_spec()
    reference = """def reference(x):
    if tuple(x.shape)==(4,3): assert not x.is_contiguous()
    if tuple(x.shape) in [(0,3),(1,3)]: assert x.stride()==(1,1)
    return x+1
"""
    if fault == "raises":
        reference = "def reference(x): raise RuntimeError('injected')\n"
    elif fault == "mutates":
        reference = "def reference(x):\n x.add_(1)\n return x\n"
    elif fault == "wrong_dtype":
        reference = "def reference(x): return x.double()\n"
    elif fault == "nan":
        reference = "def reference(x): return x*float('nan')\n"
    elif fault == "negative":
        reference = "import torch\ndef reference(x): return torch.complex(x,x).conj().imag\n"
    files["spec/reference.py"] = reference.encode()
    layout_suite = parse_recipes(files["spec/recipes.yaml"]).model_dump(mode="json")
    layout_suite["cases"][-1]["inputs"]["x"]["layout"] = "transpose"
    layout_suite["cases"].append(
        {
            "id": "singleton",
            "inputs": {
                "x": {
                    "shape": [1, 3],
                    "distribution": "arange_f32",
                    "seed": 42,
                    "layout": "transpose",
                }
            },
        }
    )
    files["spec/recipes.yaml"] = json.dumps(layout_suite).encode()
    spec = validate_spec(files, "spec")
    suite = parse_recipes(files["spec/recipes.yaml"])
    plan = {
        "files": {k: base64.b64encode(v).decode() for k, v in files.items()},
        "path": "spec",
        "spec": spec.manifest.spec.model_dump(mode="json"),
        "suite": suite.model_dump(mode="json"),
        "environment": spec.environment.model_dump(mode="json"),
    }
    tmp_path.chmod(0o755)
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    name = "cannery-m5a-oracle-test-" + uuid4().hex
    args = [
        "docker",
        "run",
        "--rm",
        "--init",
        "--pull",
        "never",
        "--name",
        name,
        "--network",
        "none",
        "--read-only",
        "--user",
        "0:0",
        "--cap-drop",
        "ALL",
        "--cap-add",
        "SETUID",
        "--cap-add",
        "SETGID",
        "--cap-add",
        "KILL",
        "--security-opt",
        "no-new-privileges",
        "--memory",
        "2g",
        "--cpus",
        "1",
        "--pids-limit",
        "64",
        "--tmpfs",
        "/work:rw,noexec,nosuid,nodev,size=32m,mode=755",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev,size=64m,mode=1777",
        "--mount",
        f"type=bind,src={tmp_path},dst=/input,readonly",
        image,
        "python",
        "-I",
        "-c",
        worker_source(),
    ]
    try:
        result = subprocess.run(args, capture_output=True, timeout=85)
        assert b"oracle reference initialized" in result.stderr, result.stderr.decode(
            errors="replace"
        )
        if fault and fault != "negative":
            assert result.returncode != 0
            assert b'"outputs"' not in result.stdout
        else:
            assert result.returncode == 0, result.stderr.decode(errors="replace")
            outputs = json.loads(result.stdout)["outputs"]
            assert set(outputs) == {c.id for c in suite.cases}
            for case in suite.cases:
                recipe = case.inputs["x"]
                data = generate(recipe, case.id, "x")
                expected = b"".join(
                    struct.pack("<f", -v if fault == "negative" else v + 1)
                    for (v,) in struct.iter_unpack("<f", data)
                )
                assert base64.b64decode(outputs[case.id]) == expected
        assert {p.name for p in tmp_path.iterdir()} == {"plan.json"}
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=15)
