"""Derive ordered native variants from the immutable spec, never author hints."""

import json
from typing import Any

from cannery.registry.tensor_manifest import ValidatedTensorSpec


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
