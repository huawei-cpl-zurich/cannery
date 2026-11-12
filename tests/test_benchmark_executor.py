import base64
import json
import tarfile
from pathlib import Path
from uuid import uuid4

import pytest
from test_build_executor import binary_package, elf
from test_build_executor import setup_executor as setup_executor
from test_numerical_evaluation import arguments as arguments
from test_recipe_spec import recipe_spec

from cannery.artifacts import digest
from cannery.benchmark.contract import PROTOCOL, Cohort
from cannery.benchmark.evidence import read_timings
from cannery.benchmark.executor import BenchmarkExecutor
from cannery.builds.executor import ExecutionFailure
from cannery.registry.manifest import validate_spec


@pytest.fixture
def benchmark(setup_executor, arguments):
    build, candidate, remote, memory = setup_executor
    baseline, suite, outputs, hashes, _, requested = arguments
    files, _, _ = recipe_spec()
    spec = validate_spec(files, "spec")
    executor = BenchmarkExecutor(
        build.profile, remote, build.snapshots, memory, build.wheel, build.wheel_sha256
    )
    cohort = Cohort(
        hardware_id="device2",
        driver="driver1",
        image=build.profile.image,
        loader_sha256=build.wheel_sha256,
        runtime=build.profile.fingerprint,
    )
    binary = memory.put(
        "binary.tar", binary_package(elf(), candidate.content_sha256, build.profile)
    )
    tensors = {
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
    payload = {
        "protocol": PROTOCOL,
        "before": tensors,
        "after": json.loads(json.dumps(tensors)),
        "timing": {
            "protocol": PROTOCOL,
            "warmup": 20,
            "cases": [{"id": c.id, "samples_us": [2.0] * 100} for c in suite.cases[:-1]],
        },
    }
    old_call, old_transfer = remote.call, remote.transfer

    def call(*args, **kwargs):
        return (
            {"content": digest(json.dumps(payload).encode())}
            if args[0] == "logs"
            else old_call(*args)
        )

    def transfer(kind, source, dest, **kwargs):
        if kind == "download":
            assert kwargs["large_output"] is True
            Path(dest).write_bytes(json.dumps(payload).encode())
        else:
            old_transfer(kind, source, dest)

    remote.call, remote.transfer = call, transfer

    def run(attempt, kind="candidate", **changes):
        options = dict(
            kind=kind,
            spec=spec,
            suite=suite,
            baseline=baseline,
            cohort=cohort,
            requested=requested,
            binary=binary,
            candidate_sha256=candidate.content_sha256,
            reference_samples={c.id: [4.0] * 100 for c in suite.cases[:-1]},
            reference_files=files,
            reference_path="spec",
        )
        return executor.benchmark(attempt, lambda: None, **(options | changes))

    return run, executor, remote, memory, payload, spec, suite, cohort


@pytest.mark.parametrize("kind", ["reference", "candidate"])
def test_output_free_evidence_and_checkpoint_recovery(benchmark, kind):
    run, executor, remote, memory, _, spec, suite, cohort = benchmark
    attempt = uuid4()
    report, artifact = run(attempt, kind)
    assert report["passed"] and report["protocol"] == PROTOCOL
    assert read_timings(memory.get(artifact), artifact, spec.content_sha256, suite, cohort) == {
        c.id: [2.0] * 100 for c in suite.cases[:-1]
    }
    assert b"input_sha256" not in memory.get(artifact) and b'"data"' not in memory.get(artifact)
    with tarfile.open(remote.journal / "benchmark.tar.gz", "r:gz") as archive:
        names = set(archive.getnames())
        assert not any("baseline" in n for n in names)
        if kind == "candidate":
            assert names == {"admitted.tar", "artifact.json", "cases.json", executor.wheel.name}
        else:
            assert "admitted.tar" not in names and any(n.startswith("tree/") for n in names)
    executor.cleanup(attempt)
    starts = remote.started.copy()
    assert run(attempt, kind) == (report, artifact) and remote.started == starts
    if kind == "candidate":
        assert report["geomean_speedup"] == 2.0


@pytest.mark.parametrize("stage", ["before", "after"])
def test_failed_correctness_never_produces_score(benchmark, stage):
    run, _, _, _, payload, _, _, _ = benchmark
    payload[stage]["cases"][0]["input_sha256"] = "0" * 64
    report, _ = run(uuid4())
    assert not report["passed"] and "geomean_speedup" not in report


def test_malformed_samples_never_get_an_accepted_report(benchmark):
    run, _, remote, _, payload, _, _, _ = benchmark
    payload["timing"]["cases"][0]["samples_us"][0] = 0
    with pytest.raises(ExecutionFailure) as result:
        run(uuid4())
    assert result.value.code == "invalid_benchmark_output"
    assert not (remote.journal / "benchmark-report.json").exists()


def test_changed_context_cannot_resume_existing_attempt(benchmark):
    run, _, remote, _, _, _, _, cohort = benchmark
    attempt = uuid4()
    run(attempt)
    starts = remote.started.copy()
    with pytest.raises(ExecutionFailure) as result:
        run(attempt, cohort=cohort.model_copy(update={"driver": "changed"}))
    assert result.value.code == "benchmark_context_changed" and remote.started == starts


@pytest.mark.parametrize("raw", [b"null", b"[]", b"{}", b"{"])
def test_corrupt_ready_timing_metadata_is_rejected_without_execution(benchmark, raw):
    _, _, remote, memory, _, spec, suite, cohort = benchmark
    artifact = memory.put("bad.json", raw)
    with pytest.raises(ValueError):
        read_timings(raw, artifact, spec.content_sha256, suite, cohort)
    assert not remote.started


def test_changed_cpu_output_bundle_cannot_reuse_passing_checkpoint(benchmark, arguments):
    from dataclasses import replace

    from cannery.numerics.baseline_format import records

    run, _, remote, _, _, _, suite, _ = benchmark
    baseline = arguments[0]
    changed = baseline.outputs | {"small": bytes(len(baseline.outputs["small"]))}
    alternate = replace(
        baseline, outputs=changed, metadata=baseline.metadata | {"cases": records(suite, changed)}
    )
    assert alternate.metadata["key"] == baseline.metadata["key"]
    attempt = uuid4()
    run(attempt)
    started = remote.started.copy()
    with pytest.raises(ExecutionFailure) as failure:
        run(attempt, baseline=alternate)
    assert failure.value.code == "benchmark_context_changed" and remote.started == started
