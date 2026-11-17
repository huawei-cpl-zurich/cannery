import hashlib
import io
import json
import tarfile

import pytest
from test_a3_contract import a3_profile
from test_build_executor import elf
from test_build_sandbox import profile
from test_tensor_spec_reader import tensor_files
from test_tensor_submission_reader import tensor_files as submission_files

from cannery.builds.executor import ExecutionFailure, binary_package
from cannery.builds.tensor_package import tensor_binary_package
from cannery.builds.tensor_signatures import tensor_variants
from cannery.registry.tensor_manifest import validate_tensor_spec
from cannery.submissions.manifest import validate_tensor_submission


def domain(tmp_path, language="triton"):
    manifest = validate_tensor_submission(submission_files(language), "candidate").manifest
    frozen = validate_tensor_spec(tensor_files(), "spec")
    return elf(), "a" * 64, profile(tmp_path), manifest, frozen


def pack(values, compiler="pinned compiler evidence"):
    return tensor_binary_package(*values, compiler=compiler)


@pytest.mark.parametrize("language", ["ascendc", "catlass", "triton"])
def test_deterministic_source_free_sdk_package(tmp_path, language):
    values = domain(tmp_path, language)
    package = pack(values)
    assert pack(values) == package
    with tarfile.open(fileobj=io.BytesIO(package), mode="r:") as archive:
        assert archive.getnames() == ["metadata.json", "kernel.so"]
        metadata = json.loads(archive.extractfile("metadata.json").read())
        assert archive.extractfile("kernel.so").read() == values[0]
    assert metadata == {
        "schema_version": 3,
        "abi": "tensor-v2",
        "device": "a5",
        "source_sha256": "a" * 64,
        "library_sha256": hashlib.sha256(elf()).hexdigest(),
        "runtime": values[2].fingerprint,
        "compiler": "pinned compiler evidence",
        "variants": tensor_variants(values[4]),
    }
    with pytest.raises(ExecutionFailure, match="tensor_package_contract_required"):
        binary_package(*values[:3], manifest=values[3])


@pytest.mark.parametrize("compiler", ["", "  ", "x" * 8193, None, 3])
def test_missing_or_unbounded_compiler_evidence_rejected(tmp_path, compiler):
    with pytest.raises(ExecutionFailure, match="compiler_evidence"):
        pack(domain(tmp_path), compiler)


@pytest.mark.parametrize(
    "binary,source",
    [
        (b"", "a" * 64),
        (b"x" * 128, "a" * 64),
        (elf(), "A" * 64),
        (elf(), None),
        (bytearray(elf()), "a" * 64),
    ],
)
def test_invalid_native_output_rejected(tmp_path, binary, source):
    values = list(domain(tmp_path))
    values[:2] = binary, source
    with pytest.raises(ExecutionFailure, match="invalid_compiled_binary"):
        pack(values)


def test_a3_profile_and_wrong_spec_version_fail_closed(tmp_path):
    values = list(domain(tmp_path))
    original = values[2]
    values[2] = a3_profile(tmp_path)
    with pytest.raises(ExecutionFailure, match="contract_mismatch"):
        pack(values)
    values[2] = original
    values[3] = values[3].model_copy(update={"spec_version": 2})
    with pytest.raises(ExecutionFailure, match="contract_mismatch"):
        pack(values)


def test_existing_unary_package_bytes_stay_identical(tmp_path):
    assert hashlib.sha256(binary_package(elf(), "a" * 64, profile(tmp_path))).hexdigest() == (
        "dde8c17d05620b30ed9aa5709355c889fa41f4520bf27c3d8281da8ea1d9ca53"
    )
