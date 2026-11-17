import copy
import io
import json
import tarfile
from pathlib import Path
from uuid import uuid4

import pytest
from test_baseline_format import rewrite
from test_build_executor import setup_executor as setup_executor
from test_tensor_numerical_executor import tensor_control as tensor_control
from test_tensor_numerical_executor import tensor_validation as tensor_validation
from test_tensor_spec_reader import tensor_files

from cannery.artifacts import digest
from cannery.benchmark.contract import Cohort
from cannery.benchmark.executor import BenchmarkExecutor
from cannery.benchmark.tensor_timing import TENSOR_PROTOCOL
from cannery.builds.executor import ExecutionFailure
from cannery.builds.remote import RemotePending
from cannery.submissions.manifest import RequestedValidation


@pytest.fixture
def tensor_benchmark(tensor_validation, tensor_control):
    _, build, remote, memory, baseline, binary = tensor_validation
    options = tensor_control["options"].copy()
    options.pop("baseline_artifact")
    executor = BenchmarkExecutor(
        build.profile, remote, build.snapshots, memory, build.wheel, build.wheel_sha256
    )
    cohort = Cohort(
        hardware_id="device4",
        driver="driver1",
        image=build.profile.image,
        loader_sha256=build.wheel_sha256,
        runtime=build.profile.fingerprint,
        protocol=TENSOR_PROTOCOL,
    )
    with tarfile.open(fileobj=io.BytesIO(memory.get(binary))) as archive:
        source = json.load(archive.extractfile("metadata.json"))["source_sha256"]
    reply = {
        "protocol": TENSOR_PROTOCOL,
        "timing": {
            "protocol": TENSOR_PROTOCOL,
            "warmup": 20,
            "cases": [{"id": "small", "samples_us": [2.0] * 100}],
        },
        "before": copy.deepcopy(tensor_control["reply"]),
        "after": copy.deepcopy(tensor_control["reply"]),
    }
    original_transfer = remote.transfer

    def raw():
        return json.dumps(reply).encode()

    def transfer(kind, source, destination, **kwargs):
        if kind == "download":
            Path(destination).write_bytes(raw())
        else:
            original_transfer(kind, source, destination)

    original_call = remote.call
    remote.transfer = transfer
    remote.call = lambda *args: (
        {"content": digest(raw())} if args[0] == "logs" else original_call(*args)
    )
    options.update(
        kind="candidate",
        candidate_sha256=source,
        cohort=cohort,
        requested=RequestedValidation(requested_rtol=0, requested_atol=0),
        reference_samples={"small": [4.0] * 100},
    )
    return executor, options, reply, remote, memory


def test_tensor_timing_is_scored_only_after_both_numeric_checks(tensor_benchmark):
    executor, options, reply, remote, memory = tensor_benchmark
    suite = options["suite"]
    options["suite"] = suite.model_copy(
        update={
            "cases": tuple(
                case.model_copy(update={"inputs": dict(reversed(list(case.inputs.items())))})
                for case in suite.cases
            )
        }
    )
    assert next(iter(options["suite"].cases[0].inputs.values())).shape == (4, 3)
    attempt = uuid4()
    report, artifact = executor.benchmark(attempt, lambda: None, **options)
    assert report["passed"] and report["protocol"] == TENSOR_PROTOCOL
    assert report["cases"][0]["shape"] == [2, 3]
    assert report["geomean_speedup"] == 2 and all(c["passed"] for c in report["correctness"])
    assert executor.benchmark(attempt, lambda: None, **options) == (report, artifact)
    assert remote.uploads.count("benchmark.tar.gz") == 1


@pytest.mark.parametrize("side", ["before", "after"])
@pytest.mark.parametrize("fault", ["readonly", "wrong_output", "alias_hash"])
def test_faulty_outputs_or_input_hashes_never_receive_scores(tensor_benchmark, side, fault):
    executor, options, reply, _, _ = tensor_benchmark
    row = reply[side]["cases"][0]
    if fault == "readonly":
        row["inputs"]["x"] = "f" * 64
    elif fault == "alias_hash":
        row["inputs"]["cache"] = "f" * 64
    else:
        row["outputs"]["indices"]["data"] = "AQAAAAEAAAA="
    report, _ = executor.benchmark(uuid4(), lambda: None, **options)
    assert not report["passed"] and "geomean_speedup" not in report


@pytest.mark.parametrize("failure", ["runtime-0", "busy"])
def test_network_disconnect_or_busy_device_resume_without_duplicate_upload(
    tensor_benchmark, failure
):
    executor, options, _, remote, _ = tensor_benchmark
    attempt = uuid4()
    remote.failure = failure
    with pytest.raises(RemotePending):
        executor.benchmark(attempt, lambda: None, **options)
    assert not (remote.journal / "benchmark-report.json").exists()
    remote.failure = None
    report, _ = executor.benchmark(attempt, lambda: None, **options)
    assert report["passed"] and remote.uploads.count("benchmark.tar.gz") == 1


def test_contract_and_cohort_mismatch_fail_before_dispatch(tensor_benchmark):
    executor, options, _, remote, _ = tensor_benchmark
    for change in [
        {"contract": None},
        {"cohort": options["cohort"].model_copy(update={"protocol": "sync-call-v1"})},
    ]:
        with pytest.raises(ExecutionFailure, match="contract_mismatch"):
            executor.benchmark(uuid4(), lambda: None, **(options | change))
    assert not remote.started


def test_binary_signature_must_match_the_frozen_tensor_spec(tensor_benchmark):
    executor, options, _, remote, memory = tensor_benchmark

    def changed(name, data):
        if name == "metadata.json":
            value = json.loads(data)
            value["variants"][0]["scalars"][0]["value"] = 2
            return json.dumps(value).encode()
        return data

    options["binary"] = memory.put(
        "bad.tar", rewrite(memory.get(options["binary"]), mutate=changed)
    )
    with pytest.raises(ExecutionFailure, match="admitted_artifact_mismatch"):
        executor.benchmark(uuid4(), lambda: None, **options)
    assert not remote.started


def test_reference_reuses_same_contract_and_records_timing_without_candidate(tensor_benchmark):
    executor, options, _, _, _ = tensor_benchmark
    options.update(kind="reference", reference_files=tensor_files(), reference_path="spec")
    for name in ["binary", "candidate_sha256", "reference_samples"]:
        options.pop(name)
    report, _ = executor.benchmark(uuid4(), lambda: None, **options)
    assert report["passed"] and "geomean_speedup" not in report
    assert report["cases"][0]["reference"]["median_us"] == 2.0
