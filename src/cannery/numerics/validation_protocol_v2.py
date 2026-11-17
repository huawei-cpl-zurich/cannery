"""Bounded named output protocol; candidate processes never receive the oracle."""

import base64
import json
import re
from dataclasses import dataclass

from cannery.numerics.recipes_v2 import RecipeSuiteV2
from cannery.numerics.tensor_contract import TensorContractV2
from cannery.registry.manifest import unique_pairs


@dataclass(frozen=True)
class TensorResultsV2:
    outputs: dict[str, dict[str, bytes]]
    input_hashes: dict[str, dict[str, str]]


def parse_tensor_results(
    data: bytes, suite: RecipeSuiteV2, contract: TensorContractV2
) -> TensorResultsV2:
    """Reject incomplete, reordered, malformed or over-budget native job replies."""
    contract.validate_suite(suite)
    if type(data) is not bytes or len(data) > 63 * 1024 * 1024:
        raise ValueError("Tensor validation output exceeds budget")
    try:
        reply = json.loads(data, object_pairs_hook=unique_pairs)
        if (
            not isinstance(reply, dict)
            or set(reply) != {"schema_version", "cases"}
            or type(reply["schema_version"]) is not int
            or reply["schema_version"] != 2
            or not isinstance(reply["cases"], list)
            or [entry["id"] for entry in reply["cases"]] != [case.id for case in suite.cases]
        ):
            raise ValueError
        outputs, hashes = {}, {}
        for case, entry in zip(suite.cases, reply["cases"], strict=True):
            definition = contract.cases[case.id]
            if (
                set(entry) != {"id", "inputs", "outputs"}
                or not isinstance(entry["inputs"], dict)
                or set(entry["inputs"]) != set(case.inputs)
                or any(
                    not isinstance(value, str) or re.fullmatch(r"[a-f0-9]{64}", value) is None
                    for value in entry["inputs"].values()
                )
                or not isinstance(entry["outputs"], dict)
                or set(entry["outputs"]) != set(definition.outputs)
            ):
                raise ValueError
            values = {}
            for name, descriptor in definition.outputs.items():
                output = entry["outputs"][name]
                if (
                    not isinstance(output, dict)
                    or set(output) != {"shape", "dtype", "data"}
                    or output["shape"] != list(descriptor.shape)
                    or not all(type(d) is int for d in output["shape"])
                    or output["dtype"] != descriptor.dtype
                    or not isinstance(output["data"], str)
                    or len(output["data"]) != 4 * ((descriptor.nbytes + 2) // 3)
                ):
                    raise ValueError
                raw = base64.b64decode(output["data"], validate=True)
                if len(raw) != descriptor.nbytes or (
                    descriptor.dtype == "bool" and any(value > 1 for value in raw)
                ):
                    raise ValueError
                values[name] = raw
            outputs[case.id], hashes[case.id] = values, entry["inputs"]
        return TensorResultsV2(outputs, hashes)
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError):
        raise ValueError("Invalid tensor validation output") from None
