import io
import json
import os
import subprocess
import tarfile
import zipfile
from dataclasses import replace
from uuid import uuid4

import pytest
import yaml
from test_build_executor import setup_executor as setup_executor
from test_build_sandbox import catlass_pins
from test_tensor_spec_reader import tensor_files
from test_tensor_submission_reader import tensor_files as submission_files

from cannery.artifacts import digest
from cannery.builds.executor import ExecutionFailure, tensor_smoke_program
from cannery.builds.remote import RemotePending
from cannery.builds.sandbox import AscendCToolchain, CatlassToolchain, TritonToolchain
from cannery.builds.tensor_signatures import tensor_variants
from cannery.git_intake import GitSource
from cannery.git_intake.snapshots import SelectedTree
from cannery.registry.tensor_manifest import validate_tensor_spec
from cannery.submissions.manifest import validate_tensor_submission


@pytest.fixture(params=["catlass", "triton", "ascendc"])
def tensor_build(setup_executor, request):
    executor, candidate, remote, memory = setup_executor
    files = submission_files(request.param)
    if request.param == "ascendc":
        value = yaml.safe_load(files["candidate/submission.yaml"])
        value["implementation"]["entrypoint"] = "factory.py:build"
        value["implementation"]["build"]["inputs"].append("factory.py")
        files["candidate/submission.yaml"] = yaml.safe_dump(value).encode()
        files["candidate/factory.py"] = b"def build(signature): pass\n"
        executor.profile = executor.profile.model_copy(
            update={"ascendc": AscendCToolchain(bisheng_sha256="d" * 64, ld_sha256="e" * 64)}
        )
    valid = validate_tensor_submission(files, "candidate")
    binding = executor.snapshots.commit(
        "d" * 64,
        GitSource(url="https://example.invalid/tensor.git", commit="b" * 40, path="candidate"),
        SelectedTree(files, dict.fromkeys(files, "100644"), "c" * 40),
    )
    candidate.snapshot = binding.model_dump(mode="json")
    candidate.manifest = valid.manifest.model_dump(mode="json")
    candidate.content_sha256, candidate.file_sha256 = valid.content_sha256, valid.file_sha256
    pins = catlass_pins() | {"package_sha256": "a" * 64, "aic_bitcode_sha256": "b" * 64}
    executor.profile = executor.profile.model_copy(update={"catlass": CatlassToolchain(**pins)})
    if request.param == "triton":
        executor.profile = executor.profile.model_copy(
            update={
                "triton": TritonToolchain(
                    cube_cores=32,
                    vector_cores=64,
                    version="3.6.0",
                    package_root="/home/operator/triton-toolchain/triton",
                    package_sha256="a" * 64,
                    bishengir_sha256="b" * 64,
                    bisheng_sha256="c" * 64,
                )
            }
        )
    with zipfile.ZipFile(executor.wheel, "w") as archive:
        archive.writestr("cannery_kernels/tensor_abi.h", "int cannery_prepare_tensor_v2(void);")
    executor.wheel_sha256 = digest(executor.wheel.read_bytes())
    return executor, candidate, remote, memory, validate_tensor_spec(tensor_files(), "spec")


def test_tensor_compile_package_and_native_smoke_follow_durable_flow(tensor_build):
    executor, candidate, remote, memory, frozen = tensor_build
    attempt = uuid4()
    result = executor.execute(candidate, attempt, lambda: None, frozen=frozen)
    with tarfile.open(fileobj=io.BytesIO(memory.get(result))) as archive:
        metadata = json.load(archive.extractfile("metadata.json"))
    assert metadata["schema_version"] == 3 and metadata["abi"] == "tensor-v2"
    assert metadata["variants"] == tensor_variants(frozen)
    compiler = json.loads(metadata["compiler"])
    backend = candidate.manifest["implementation"]["language"]
    if backend == "ascendc":
        assert compiler[backend] == {"bisheng_sha256": "d" * 64, "ld_sha256": "e" * 64}
    else:
        assert compiler[backend]["package_sha256"] == "a" * 64
    assert "bitcode_root" not in compiler[backend] and "package_root" not in compiler[backend]
    context = json.loads((remote.journal / "context.json").read_bytes())
    assert context["tensor_spec"] == frozen.spec.content_sha256
    assert remote.started == {"prepare", "unpack-source", "compile", "unpack-runtime", "runtime-0"}
    assert executor.execute(candidate, attempt, lambda: None, frozen=frozen) == result
    assert remote.uploads.count("source.tar.gz") == 1


@pytest.mark.parametrize("failure", ["compile", "runtime-0", "busy"])
def test_tensor_lost_reply_and_device_busy_resume_same_build(tensor_build, failure):
    executor, candidate, remote, memory, frozen = tensor_build
    remote.failure = failure
    attempt = uuid4()
    with pytest.raises(RemotePending):
        executor.execute(candidate, attempt, lambda: None, frozen=frozen)
    assert not (remote.journal / "admission.json").exists()
    remote.failure = None
    executor.execute(candidate, attempt, lambda: None, frozen=frozen)
    assert remote.uploads.count("source.tar.gz") == 1


def test_tensor_spec_identity_change_fences_retained_build(tensor_build):
    executor, candidate, remote, memory, frozen = tensor_build
    remote.failure = "compile"
    attempt = uuid4()
    with pytest.raises(RemotePending):
        executor.execute(candidate, attempt, lambda: None, frozen=frozen)
    changed = replace(frozen, spec=replace(frozen.spec, content_sha256="f" * 64))
    with pytest.raises(ExecutionFailure, match="worker_profile_changed"):
        executor.execute(candidate, attempt, lambda: None, frozen=changed)


def test_tensor_requires_pinned_header_before_remote_dispatch(tensor_build):
    executor, candidate, remote, memory, frozen = tensor_build
    with zipfile.ZipFile(executor.wheel, "w") as archive:
        archive.writestr("other", "missing tensor ABI")
    executor.wheel_sha256 = digest(executor.wheel.read_bytes())
    with pytest.raises(ExecutionFailure, match="tensor_loader_header_missing"):
        executor.execute(candidate, uuid4(), lambda: None, frozen=frozen)
    assert not remote.started and not memory.puts


def test_smoke_regenerates_bounded_integer_inputs_in_isolated_cpu_simulation(tmp_path):
    image = os.environ.get("CANNERY_ORACLE_TEST_IMAGE")
    if not image:
        pytest.skip("Set CANNERY_ORACLE_TEST_IMAGE for isolated CPU smoke simulation")
    from test_tensor_oracle_worker import REFERENCE, plan

    from cannery.numerics.oracle_job import CPU_FLAGS
    from cannery.numerics.recipes_v2 import RecipeSuiteV2, generate_v2, input_sha256_v2

    suite, request = plan(REFERENCE)
    for case in request["suite"]["cases"]:
        case["inputs"]["indices"]["low"] = 1
    suite = RecipeSuiteV2.model_validate(request["suite"])
    request["input_sha256"] = {
        case.id: {
            name: input_sha256_v2(recipe, generate_v2(recipe, case.id, name))
            for name, recipe in case.inputs.items()
        }
        for case in suite.cases
    }
    tmp_path.chmod(0o755)
    (tmp_path / "cases.json").write_text(json.dumps(request))
    reference = REFERENCE.replace(
        "    assert os.geteuid()==10001",
        '    assert bool((indices>=1).all()), "smoke violated frozen integer bounds"',
    )
    source = tensor_smoke_program().split("# Reserve the result channel", 1)[0]
    source += """
import torch
wire=sys.stdout.buffer
torch_npu=SimpleNamespace(get_npu_format=lambda _:2,
    empty_with_format=lambda shape,dtype,**_:torch.empty(shape,dtype=dtype))
torch.npu=SimpleNamespace(synchronize=lambda:None)
"""
    source += (
        reference
        + """
class Kernel:
    metadata={'schema_version':3,'abi':'tensor-v2', 'variants':[{'inputs':[
        {'name':n} for n in ['x','cache','indices','mask']]}]}
    forward=staticmethod(reference)
kernel=Kernel()
"""
    )
    # Execute the complete smoke loop, with only the device/bootstrap simulated.
    program = tensor_smoke_program()
    source += program[program.index("plan=json.loads") :]
    name = "cannery-llm-catalog-smoke-bounds-" + uuid4().hex
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
        assert result.returncode == 0, result.stderr.decode(errors="replace")
        assert (
            result.stdout == b"tensor binary loaded and called; "
            b"numerical correctness and benchmarks remain pending"
        )
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=15)
