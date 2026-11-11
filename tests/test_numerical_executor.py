import base64
import json
import tarfile
from uuid import uuid4

import pytest
from test_build_executor import binary_package, elf
from test_build_executor import setup_executor as setup_executor
from test_numerical_evaluation import arguments as arguments

from cannery.artifacts import digest
from cannery.builds.executor import ExecutionFailure
from cannery.builds.remote import RemoteFailure, RemotePending
from cannery.numerics.executor import NumericalExecutor


@pytest.fixture
def validation(setup_executor, arguments):
    from test_recipe_spec import recipe_spec

    from cannery.registry.manifest import validate_spec

    build, candidate, remote, memory = setup_executor
    baseline, suite, outputs, hashes, _, _ = arguments
    spec = validate_spec(recipe_spec()[0], "spec")
    executor = NumericalExecutor(
        build.profile, remote, build.snapshots, memory, build.wheel, build.wheel_sha256
    )
    binary = memory.put(
        "binary.tar", binary_package(elf(), candidate.content_sha256, build.profile)
    )
    reference = memory.put("baseline.tar", b"private cached oracle outputs")
    raw = json.dumps(
        {
            "schema_version": 1,
            "cases": [
                {
                    "id": c.id,
                    "shape": list(c.inputs["x"].shape),
                    "dtype": "float32",
                    "input_sha256": hashes[c.id],
                    "data": base64.b64encode(outputs[c.id]).decode(),
                }
                for c in suite.cases
            ],
        }
    ).encode()
    original = remote.call

    def call(*args):
        return {"content": digest(raw)} if args[0] == "logs" else original(*args)

    remote.call = call
    original_transfer = remote.transfer

    def transfer(kind, source, dest):
        if kind == "download":
            from pathlib import Path

            Path(dest).write_bytes(raw)
        else:
            original_transfer(kind, source, dest)

    remote.transfer = transfer

    def run(attempt):
        return executor.validate(
            candidate,
            attempt,
            lambda: None,
            spec=spec,
            suite=suite,
            baseline=baseline,
            binary=binary,
            baseline_artifact=reference,
        )

    return run, executor, remote, memory, baseline, binary


def test_private_baseline_outputs_never_uploaded_and_report_survives_restart(validation):
    run, executor, remote, _, baseline, _ = validation
    attempt = uuid4()
    report = run(attempt)
    assert report["passed"] and len(report["cases"]) == 3
    with tarfile.open(remote.journal / "validation.tar.gz", "r:gz") as archive:
        names = set(archive.getnames())
        assert names == {"admitted.tar", "artifact.json", executor.wheel.name, "cases.json"}
        plan = json.load(archive.extractfile("cases.json"))
        assert set(plan) == {"suite", "input_sha256"}
        assert all("output_sha256" not in value for value in plan["suite"]["cases"])
    executor.cleanup(attempt)
    started = remote.started.copy()
    assert run(attempt) == report and remote.started == started


def test_transport_disconnect_resumes_same_attempt(validation):
    run, _, remote, _, _, _ = validation
    remote.failure = "runtime-0"
    attempt = uuid4()
    with pytest.raises(RemotePending):
        run(attempt)
    assert run(attempt)["passed"]
    assert remote.uploads.count("validation.tar.gz") == 1


def test_busy_device_reprobes_without_new_upload_or_cpu_reference(validation):
    run, _, remote, _, _, _ = validation
    remote.failure = "busy"
    attempt = uuid4()
    with pytest.raises(RemotePending):
        run(attempt)
    assert run(attempt)["passed"]
    assert {"runtime-0", "runtime-1"} <= remote.started
    assert remote.uploads == ["validation.tar.gz"]


def test_corrupt_output_digest_fails_before_comparison(validation):
    run, _, remote, _, _, _ = validation
    remote.call = lambda *args: {"content": "a" * 64}
    with pytest.raises(ExecutionFailure, match="numerical_digest_mismatch"):
        run(uuid4())


def test_unhealthy_runtime_failure_is_bounded_retryable(validation):
    run, _, remote, _, _, _ = validation
    original = remote.operation

    def operation(name, *args, **kwargs):
        if name.startswith("runtime-"):
            raise RemoteFailure(1)
        return original(name, *args, **kwargs)

    remote.operation = operation
    with pytest.raises(ExecutionFailure) as failure:
        run(uuid4())
    assert failure.value.retryable and failure.value.code == "numerical_runtime_failed"
