"""Package a verified native tensor library using its frozen spec call domain."""

from cannery.artifacts import digest
from cannery.builds.library import ExecutionFailure, archive_library, check_library
from cannery.builds.sandbox import WorkerProfile
from cannery.builds.tensor_signatures import tensor_variants
from cannery.registry.tensor_manifest import ValidatedTensorSpec
from cannery.submissions.manifest import SubmissionManifest


def tensor_binary_package(
    binary: bytes,
    source: str,
    profile: WorkerProfile,
    manifest: SubmissionManifest,
    frozen: ValidatedTensorSpec,
    *,
    compiler: str,
) -> bytes:
    """Compiler evidence comes from the trusted build controller, never source output.

    This writer is standalone until build admission supports the complete tensor
    pipeline. Only the ELF and frozen static signatures enter the runtime package.
    """
    if (
        manifest.schema_version != 2
        or manifest.implementation.build.abi != "tensor-v2"
        or manifest.implementation.device != "a5"
        or profile.device != "a5"
        or frozen.spec.manifest.schema_version != 2
        or manifest.spec_version != frozen.spec.manifest.spec.version
    ):
        raise ExecutionFailure("tensor_package_contract_mismatch")
    if type(compiler) is not str or not compiler.strip() or len(compiler) > 8192:
        raise ExecutionFailure("tensor_compiler_evidence_required")
    check_library(binary, source)
    # Re-read the frozen model domain rather than trusting build-produced metadata.
    metadata = {
        "schema_version": 3,
        "abi": "tensor-v2",
        "device": "a5",
        "source_sha256": source,
        "library_sha256": digest(binary),
        "runtime": profile.fingerprint,
        "compiler": compiler,
        "variants": tensor_variants(frozen),
    }
    encoded = archive_library(binary, metadata, metadata_limit=256 * 1024)
    if len(encoded) > 64 * 1024 * 1024:
        raise ExecutionFailure("tensor_package_size_limit")
    return encoded
