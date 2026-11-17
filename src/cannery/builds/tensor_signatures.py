"""Derive ordered native variants from the immutable spec, never author hints."""

import json
from typing import Any

from cannery.builds.ascendc import BuildError
from cannery.registry.tensor_manifest import ValidatedTensorSpec
from cannery.submissions.manifest import SubmissionManifest


def tensor_variants(frozen: ValidatedTensorSpec) -> list[dict[str, Any]]:
    """Collapse seed-only cases while preserving exact scalar and mutation domains."""
    frozen.contract.validate_suite(frozen.suite)
    interface = frozen.spec.manifest.spec.interface
    variants, domains = [], set()
    for case in frozen.suite.cases:
        definition = frozen.contract.cases[case.id]
        if set(case.inputs) != set(interface.inputs) or set(definition.outputs) != set(
            interface.outputs
        ):
            raise ValueError("Tensor interface differs from the frozen spec")
        variant = {
            "inputs": [
                {
                    "name": name,
                    "shape": list(case.inputs[name].shape),
                    "dtype": case.inputs[name].dtype,
                }
                for name in interface.inputs
            ],
            "outputs": [
                {
                    "name": name,
                    "shape": list(definition.outputs[name].shape),
                    "dtype": definition.outputs[name].dtype,
                    "alias": definition.aliases.get(name),
                }
                for name in interface.outputs
            ],
            "scalars": [
                {"name": name, **definition.scalars[name].model_dump(mode="json")}
                for name in sorted(definition.scalars)
            ],
        }
        domain = json.dumps({k: variant[k] for k in ("inputs", "scalars")}, sort_keys=True)
        if domain not in domains:
            domains.add(domain)
            variants.append(variant)
    return variants


def tensor_build_parameters(
    manifest: SubmissionManifest, frozen: ValidatedTensorSpec, abi_header: str, language: str
) -> dict[str, Any]:
    impl = manifest.implementation
    if (
        (manifest.schema_version, impl.language, impl.device, impl.build.abi)
        != (2, language, "a5", "tensor-v2")
        or impl.build.options.optimization != "release"
        or frozen.spec.manifest.schema_version != 2
        or manifest.spec_version != frozen.spec.manifest.spec.version
        or manifest.kernel
        != f"{frozen.spec.manifest.kernel.namespace}/{frozen.spec.manifest.kernel.name}"
        or type(abi_header) is not str
        or "cannery_prepare_tensor_v2" not in abi_header
        or len(abi_header.encode()) > 16384
    ):
        raise BuildError("Requires matching A5 tensor spec, submission and pinned ABI header")
    entry, symbol = impl.entrypoint.split(":")
    return {
        "entry": entry,
        "symbol": symbol,
        "variants": tensor_variants(frozen),
        "abi_header": abi_header,
    }
