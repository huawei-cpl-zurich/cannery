import json
import os
import struct
import subprocess
import time
from uuid import uuid4

import pytest
from test_build_executor import Memory
from test_tensor_spec_reader import tensor_files

from cannery.artifacts import ArtifactError
from cannery.numerics.baseline_v2 import baseline_key_v2, unpack_baseline_v2
from cannery.numerics.oracle_job import OracleJob, OraclePending
from cannery.numerics.recipes_v2 import generate_v2
from cannery.registry.tensor_manifest import validate_tensor_spec


def test_real_tensor_oracle_resumes_after_transport_and_upload_loss(tmp_path, monkeypatch):
    image = os.environ.get("CANNERY_ORACLE_TEST_IMAGE")
    if not image:
        pytest.skip("Set CANNERY_ORACLE_TEST_IMAGE for retained tensor oracle Docker proof")
    from cannery.numerics import oracle_job

    files = tensor_files()
    files["spec/reference.py"] = b"""import torch
def reference(x,cache,offset):
    cache[:2].copy_(x)
    return {'updated':cache,'indices':torch.arange(2,dtype=torch.int32)+offset}
"""
    frozen = validate_tensor_spec(files, "spec")
    spec, suite, contract = frozen.spec, frozen.suite, frozen.contract.output_contract()
    key = baseline_key_v2(spec.content_sha256, suite, spec.environment, contract)
    attempt = uuid4()
    volume = "cannery-llm-catalog-oracle-" + attempt.hex
    image_id = subprocess.check_output(
        ["docker", "image", "inspect", image, "--format", "{{.Id}}"], text=True, timeout=20
    ).strip()
    subprocess.run(
        [
            "docker",
            "volume",
            "create",
            "--label",
            "cannery.task=llm-catalog",
            "--driver",
            "local",
            "--opt",
            "type=none",
            "--opt",
            f"device={tmp_path}",
            "--opt",
            "o=bind",
            volume,
        ],
        check=True,
        capture_output=True,
        timeout=20,
    )
    calls, lost = [], set()
    original = oracle_job.docker

    def unreliable(arguments, limit=1024 * 1024):
        result = original(arguments, limit)
        calls.append(arguments[0])
        if arguments[0] in {"create", "start"} and arguments[0] not in lost:
            lost.add(arguments[0])
            raise OraclePending("injected lost Docker response")
        return result

    monkeypatch.setattr(oracle_job, "docker", unreliable)
    memory = Memory()
    put = memory.put

    def disconnect(*args):
        if "upload" not in lost:
            lost.add("upload")
            raise ArtifactError("injected artifact network disconnect")
        return put(*args)

    memory.put = disconnect

    def job():
        return OracleJob(tmp_path, volume, key, attempt, image_id, memory, lambda: None)

    try:
        deadline = time.monotonic() + 90
        while True:
            try:
                reference = job().progress(spec, suite, files, "spec")
                break
            except (OraclePending, ArtifactError):
                assert time.monotonic() < deadline, calls
                time.sleep(0.2)
        assert lost == {"create", "start", "upload"}
        assert calls.count("create") == calls.count("start") == 1
        assert job().progress(spec, suite, files, "spec") == reference
        bundle = unpack_baseline_v2(
            memory.get(reference),
            reference.sha256,
            spec_sha256=spec.content_sha256,
            suite=suite,
            environment=spec.environment,
            contract=contract,
        )
        case = suite.cases[0]
        x = generate_v2(case.inputs["x"], case.id, "x")
        cache = generate_v2(case.inputs["cache"], case.id, "cache")
        assert bundle.outputs[case.id] == {
            "updated": x + cache[len(x) :],
            "indices": struct.pack("<2i", 1, 2),
        }
        assert not (job().journal / "output/result.json").exists()
        assert not (job().journal / "input/plan.json").exists()
        assert (
            json.loads((job().journal / "artifact.json").read_bytes())["sha256"] == reference.sha256
        )
    finally:
        job().cleanup()
        subprocess.run(
            ["docker", "volume", "rm", volume], check=True, capture_output=True, timeout=20
        )
