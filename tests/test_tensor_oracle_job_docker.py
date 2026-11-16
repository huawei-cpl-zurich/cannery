import json
import os
import struct
import subprocess
import time
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from test_build_executor import Memory
from test_tensor_spec_reader import tensor_files

from cannery.artifacts import ArtifactError
from cannery.database import Base
from cannery.git_intake import GitSource
from cannery.git_intake.snapshots import SelectedTree, SnapshotStore
from cannery.numerics.baseline_state import BaselineState, ensure_baseline
from cannery.numerics.baseline_v2 import baseline_key_v2, unpack_baseline_v2
from cannery.numerics.baselines import BaselinePending, advance_baseline
from cannery.numerics.oracle_job import OracleJob, OraclePending
from cannery.numerics.recipes_v2 import generate_v2
from cannery.registry.freeze import freeze_spec, kernel_key
from cannery.registry.tensor_manifest import validate_tensor_spec


@pytest.mark.parametrize("coordinated", [False, True])
def test_real_tensor_oracle_resumes_after_transport_and_upload_loss(
    tmp_path, monkeypatch, coordinated
):
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
    engine = create_engine("sqlite:///" + str(tmp_path / "db")) if coordinated else None
    snapshots = SnapshotStore(tmp_path / "snapshots")
    binding = None
    if engine is not None:
        Base.metadata.create_all(engine)
        binding = snapshots.commit(
            kernel_key("team", "identity"),
            GitSource(url="https://example.invalid/spec.git", commit="a" * 40, path="spec"),
            SelectedTree(files, dict.fromkeys(files, "100644"), "b" * 40),
        )
        ensure_baseline(engine, key, spec.content_sha256, binding)
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
        current = attempt
        if engine is not None:
            with Session(engine) as db:
                current = db.get(BaselineState, key).attempt or current
        return OracleJob(tmp_path, volume, key, current, image_id, memory, lambda: None)

    def progress():
        if engine is None:
            return job().progress(spec, suite, files, "spec")
        return advance_baseline(
            engine,
            key,
            spec,
            suite,
            contract=frozen.contract,
            snapshots=snapshots,
            artifacts=memory,
            root=tmp_path,
            volume=volume,
            image=image_id,
            fence=lambda: None,
        ).artifact

    try:
        deadline = time.monotonic() + 90
        while True:
            try:
                reference = progress()
                break
            except (OraclePending, ArtifactError, BaselinePending):
                assert time.monotonic() < deadline, calls
                if engine is not None:
                    with Session(engine) as db, db.begin():
                        row = db.get(BaselineState, key)
                        row.due_at = row.lease_until = datetime.now(UTC) - timedelta(seconds=1)
                time.sleep(0.2)
        assert lost == {"create", "start", "upload"}
        assert calls.count("create") == calls.count("start") == 1
        assert progress() == reference
        if engine is not None:
            evidence = advance_baseline(
                engine,
                key,
                spec,
                suite,
                contract=frozen.contract,
                snapshots=snapshots,
                artifacts=memory,
                root=tmp_path,
                volume=volume,
                image=image_id,
                fence=lambda: None,
            )
            version = freeze_spec(
                engine,
                spec,
                binding,
                evidence,
                store=snapshots,
                namespace="team",
                allow_new_version=False,
            )
            assert version.manifest["schema_version"] == 2
            with Session(engine) as db:
                assert db.get(BaselineState, key).attempts == 1
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
        if engine is not None:
            engine.dispose()
        subprocess.run(
            ["docker", "volume", "rm", volume], check=True, capture_output=True, timeout=20
        )
