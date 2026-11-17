import base64
import json
import tarfile
from pathlib import Path
from uuid import uuid4

import pytest
import test_numerical_executor as legacy
from test_baseline_format import rewrite
from test_build_executor import binary_package, elf
from test_build_executor import setup_executor as setup_executor
from test_numerical_evaluation import arguments as arguments
from test_tensor_spec_reader import tensor_files
from test_tensor_submission_reader import tensor_files as submission_files

from cannery.artifacts import digest
from cannery.builds.executor import ExecutionFailure
from cannery.builds.tensor_signatures import tensor_variants
from cannery.numerics.baseline_v2 import pack_baseline_v2, unpack_baseline_v2
from cannery.numerics.executor import NumericalExecutor
from cannery.numerics.recipes_v2 import input_sha256_v2
from cannery.registry.tensor_manifest import validate_tensor_spec
from cannery.submissions.manifest import validate_tensor_submission

unary_validation = legacy.validation


@pytest.fixture
def tensor_control():
    return {}


@pytest.fixture
def tensor_validation(setup_executor, tensor_control):
    build, candidate, remote, memory = setup_executor
    declared = validate_tensor_submission(submission_files(), "candidate")
    candidate.manifest = declared.manifest.model_dump(mode="json")
    candidate.content_sha256 = declared.content_sha256
    frozen = validate_tensor_spec(tensor_files(), "spec")
    spec, suite, contract = frozen.spec, frozen.suite, frozen.contract
    outputs = {
        case: {name: bytes(value.nbytes) for name, value in definition.outputs.items()}
        for case, definition in contract.cases.items()
    }
    package = pack_baseline_v2(
        spec.content_sha256,
        suite,
        spec.environment,
        "sha256:" + "a" * 64,
        contract.output_contract(),
        outputs,
    )
    baseline = unpack_baseline_v2(
        package,
        digest(package),
        spec_sha256=spec.content_sha256,
        suite=suite,
        environment=spec.environment,
        contract=contract.output_contract(),
    )
    reference = memory.put("baseline.tar", package)

    def metadata(name, data):
        if name != "metadata.json":
            return data
        record = json.loads(data)
        record.update(schema_version=3, abi="tensor-v2", variants=tensor_variants(frozen))
        return json.dumps(record).encode()

    binary = memory.put(
        "binary.tar",
        rewrite(binary_package(elf(), candidate.content_sha256, build.profile), mutate=metadata),
    )
    reply = {
        "schema_version": 2,
        "cases": [
            {
                "id": case.id,
                "inputs": dict(baseline.metadata["cases"][index]["inputs"]),
                "outputs": {
                    name: {
                        "shape": list(value.shape),
                        "dtype": value.dtype,
                        "data": base64.b64encode(outputs[case.id][name]).decode(),
                    }
                    for name, value in contract.cases[case.id].outputs.items()
                },
            }
            for index, case in enumerate(suite.cases)
        ],
    }
    reply["cases"][0]["inputs"]["cache"] = input_sha256_v2(
        suite.cases[0].inputs["cache"], outputs["small"]["updated"]
    )
    tensor_control.update(reply=reply)

    def raw():
        return json.dumps(tensor_control["reply"]).encode()

    original = remote.call
    remote.call = lambda *args: {"content": digest(raw())} if args[0] == "logs" else original(*args)
    original_transfer = remote.transfer

    def transfer(kind, source, destination):
        if kind == "download":
            Path(destination).write_bytes(raw())
        else:
            original_transfer(kind, source, destination)

    remote.transfer = transfer
    executor = NumericalExecutor(
        build.profile, remote, build.snapshots, memory, build.wheel, build.wheel_sha256
    )
    options = dict(
        spec=spec,
        suite=suite,
        baseline=baseline,
        binary=binary,
        baseline_artifact=reference,
        contract=contract,
    )
    tensor_control["options"] = options

    def run(attempt):
        return executor.validate(candidate, attempt, lambda: None, **options)

    return run, executor, remote, memory, baseline, binary


@pytest.fixture(params=["unary_validation", "tensor_validation"])
def scenario(request):
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize(
    "check",
    [
        legacy.test_transport_disconnect_resumes_same_attempt,
        legacy.test_busy_device_reprobes_without_new_upload_or_cpu_reference,
        legacy.test_corrupt_output_digest_fails_before_comparison,
        legacy.test_unhealthy_runtime_failure_is_bounded_retryable,
    ],
)
def test_shared_lifecycle_faults(scenario, check):
    check(scenario)


def test_named_result_mutation_cached_report_and_private_oracle(tensor_validation):
    run, executor, remote, _, _, _ = tensor_validation
    attempt = uuid4()
    report = run(attempt)
    assert report["passed"] and report["schema_version"] == 2
    case = report["cases"][0]
    assert not case["inputs"]["cache"]["unchanged"] and case["inputs"]["cache"]["mutable"]
    assert case["readonly_inputs_unchanged"]
    with tarfile.open(remote.journal / "validation.tar.gz", "r:gz") as archive:
        assert set(archive.getnames()) == {
            "admitted.tar",
            "artifact.json",
            executor.wheel.name,
            "cases.json",
        }
        plan = json.load(archive.extractfile("cases.json"))
        assert set(plan) == {"suite", "contract", "input_sha256"}
        assert "data" not in json.dumps(plan) and "sha256" not in json.dumps(plan["contract"])
    executor.cleanup(attempt)
    started = remote.started.copy()
    assert run(attempt) == report and remote.started == started


def test_readonly_corruption_produces_failed_report(tensor_validation, tensor_control):
    tensor_control["reply"]["cases"][0]["inputs"]["x"] = "0" * 64
    report = tensor_validation[0](uuid4())
    assert not report["passed"] and not report["cases"][0]["readonly_inputs_unchanged"]


def test_truncated_named_output_fails_closed(tensor_validation, tensor_control):
    tensor_control["reply"]["cases"][0]["outputs"]["updated"]["data"] = ""
    with pytest.raises(ExecutionFailure, match="invalid_numerical_output"):
        tensor_validation[0](uuid4())


def test_missing_contract_rejected_before_device_job(tensor_validation, tensor_control):
    tensor_control["options"]["contract"] = None
    with pytest.raises(ExecutionFailure, match="tensor_validation_contract_mismatch"):
        tensor_validation[0](uuid4())
    assert not tensor_validation[2].started


def test_changed_tensor_contract_cannot_reuse_completed_report(tensor_validation, tensor_control):
    run = tensor_validation[0]
    attempt = uuid4()
    assert run(attempt)["passed"]
    contract = tensor_control["options"]["contract"]
    definition = contract.cases["small"]
    tensor_control["options"]["contract"] = contract.model_copy(
        update={"cases": {"small": definition.model_copy(update={"aliases": {}})}}
    )
    with pytest.raises(ExecutionFailure, match="validation_context_changed"):
        run(attempt)


@pytest.mark.parametrize("kind", ["abi", "domain"])
def test_binary_domain_mismatch_rejected_before_device_job(tensor_validation, tensor_control, kind):
    run, _, remote, memory, _, binary = tensor_validation

    def mutate(name, data):
        if name != "metadata.json":
            return data
        metadata = json.loads(data)
        if kind == "abi":
            metadata["abi"] = "unary-f32-v1"
        else:
            metadata["variants"][0]["inputs"][0]["shape"] = [1]
        return json.dumps(metadata).encode()

    tensor_control["options"]["binary"] = memory.put(
        "bad.tar", rewrite(memory.get(binary), mutate=mutate)
    )
    with pytest.raises(ExecutionFailure, match="admitted_artifact_mismatch"):
        run(uuid4())
    assert not remote.started


def test_mutable_hash_and_output_alias_mismatch_produces_failed_report(
    tensor_validation, tensor_control
):
    tensor_control["reply"]["cases"][0]["inputs"]["cache"] = "0" * 64
    report = tensor_validation[0](uuid4())
    assert not report["passed"] and not report["cases"][0]["mutable_aliases_consistent"]
