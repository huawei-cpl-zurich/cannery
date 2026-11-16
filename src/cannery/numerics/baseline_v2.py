"""Bounded cached multi-output oracle bundles; never pickle or author paths.

Standalone reader/writer primitives. Registration continues to require v1 until
the trusted reference supervisor and device worker both implement this format.
"""

import hashlib
import io
import json
import math
import tarfile
from dataclasses import dataclass
from typing import Any, Self

from pydantic import Field, field_validator, model_validator

from cannery.numerics.recipes import MAX_SUITE_BYTES, MAX_TENSOR_ELEMENTS, Dimension
from cannery.numerics.recipes_v2 import (
    ITEM_BYTES,
    DType,
    RecipeSuiteV2,
    generate_v2,
    input_sha256_v2,
)
from cannery.registry.manifest import EnvironmentLock, Name, StrictModel

MAX_METADATA_BYTES = 256 * 1024
MAX_PACKAGE_BYTES = MAX_SUITE_BYTES + 2 * 1024 * 1024


class OutputTensorV2(StrictModel):
    shape: tuple[Dimension, ...] = Field(min_length=1, max_length=4)
    dtype: DType

    @field_validator("shape", mode="before")
    @classmethod
    def dimensions(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def bounded(self) -> Self:
        if math.prod(self.shape) > MAX_TENSOR_ELEMENTS:
            raise ValueError("Output exceeds element budget")
        return self

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * ITEM_BYTES[self.dtype]


class OutputContractV2(StrictModel):
    cases: dict[Name, dict[Name, OutputTensorV2]] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def bounded(self) -> Self:
        if any(not 1 <= len(outputs) <= 32 for outputs in self.cases.values()):
            raise ValueError("Expected one to 32 named outputs per case")
        if (
            sum(t.nbytes for outputs in self.cases.values() for t in outputs.values())
            > MAX_SUITE_BYTES
        ):
            raise ValueError("Baseline exceeds aggregate output budget")
        return self


@dataclass(frozen=True)
class BaselineBundleV2:
    metadata: dict[str, Any]
    outputs: dict[str, dict[str, bytes]]


def _digest(value: str) -> bool:
    return len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def baseline_key_v2(
    spec_sha256: str,
    suite: RecipeSuiteV2,
    environment: EnvironmentLock,
    contract: OutputContractV2,
) -> str:
    if not _digest(spec_sha256):
        raise ValueError("Invalid frozen spec identity")
    if set(contract.cases) != {case.id for case in suite.cases}:
        raise ValueError("Output contract must cover every recipe case")
    value = {
        "spec_sha256": spec_sha256,
        "suite_sha256": suite.sha256,
        "generator": suite.generator,
        "environment": environment.model_dump(mode="json"),
        "outputs": contract.model_dump(mode="json"),
    }
    return hashlib.sha256(_json(value)).hexdigest()


def _json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _metadata(
    spec_sha256: str,
    suite: RecipeSuiteV2,
    environment: EnvironmentLock,
    image_id: str,
    contract: OutputContractV2,
    outputs: dict[str, dict[str, bytes]],
) -> bytes:
    key = baseline_key_v2(spec_sha256, suite, environment, contract)
    if not image_id.startswith("sha256:") or not _digest(image_id[7:]):
        raise ValueError("Expected immutable oracle image ID")
    if set(outputs) != set(contract.cases):
        raise ValueError("Baseline must cover every case exactly once")
    cases = []
    for case in suite.cases:
        if set(outputs[case.id]) != set(contract.cases[case.id]):
            raise ValueError("Baseline must cover every named output exactly once")
        records = {}
        for name, descriptor in contract.cases[case.id].items():
            data = outputs[case.id][name]
            if type(data) is not bytes or len(data) != descriptor.nbytes:
                raise ValueError("Reference output byte length/dtype/shape mismatch")
            if descriptor.dtype == "bool" and any(v > 1 for v in data):
                raise ValueError("Boolean output must use canonical bytes")
            records[name] = descriptor.model_dump(mode="json") | {
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        cases.append(
            {
                "id": case.id,
                "inputs": {
                    name: input_sha256_v2(recipe, generate_v2(recipe, case.id, name))
                    for name, recipe in case.inputs.items()
                },
                "outputs": records,
            }
        )
    encoded = _json(
        {
            "schema_version": 2,
            "key": key,
            "spec_sha256": spec_sha256,
            "suite_sha256": suite.sha256,
            "generator": suite.generator,
            "environment": environment.model_dump(mode="json"),
            "image_id": image_id,
            "cases": cases,
        }
    )
    if len(encoded) > MAX_METADATA_BYTES:
        raise ValueError("Baseline metadata exceeds budget")
    return encoded


def pack_baseline_v2(
    spec_sha256: str,
    suite: RecipeSuiteV2,
    environment: EnvironmentLock,
    image_id: str,
    contract: OutputContractV2,
    outputs: dict[str, dict[str, bytes]],
) -> bytes:
    encoded = _metadata(spec_sha256, suite, environment, image_id, contract, outputs)
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        members = [("metadata.json", encoded)] + [
            (f"outputs/{case}/{name}.bin", outputs[case][name])
            for case in sorted(contract.cases)
            for name in sorted(contract.cases[case])
        ]
        for name, data in members:
            member = tarfile.TarInfo(name)
            member.size, member.mode = len(data), 0o400
            archive.addfile(member, io.BytesIO(data))
    package = stream.getvalue()
    if len(package) > MAX_PACKAGE_BYTES:
        raise ValueError("Baseline package exceeds budget")
    return package


def unpack_baseline_v2(
    package: bytes,
    sha256: str,
    *,
    spec_sha256: str,
    suite: RecipeSuiteV2,
    environment: EnvironmentLock,
    contract: OutputContractV2,
) -> BaselineBundleV2:
    if (
        type(package) is not bytes
        or len(package) > MAX_PACKAGE_BYTES
        or hashlib.sha256(package).hexdigest() != sha256
    ):
        raise ValueError("Baseline digest or size mismatch")
    baseline_key_v2(spec_sha256, suite, environment, contract)
    limits = {"metadata.json": MAX_METADATA_BYTES} | {
        f"outputs/{case}/{name}.bin": descriptor.nbytes
        for case, outputs in contract.cases.items()
        for name, descriptor in outputs.items()
    }
    try:
        contents = {}
        with tarfile.open(fileobj=io.BytesIO(package), mode="r:") as archive:
            for member in archive:
                if member.name not in limits or member.name in contents or not member.isfile():
                    raise ValueError("Unexpected or duplicate baseline member")
                limit = limits[member.name]
                if not 0 <= member.size <= limit:
                    raise ValueError("Baseline member exceeds budget")
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError("Missing baseline member")
                contents[member.name] = source.read(limit + 1)
                if len(contents[member.name]) != member.size:
                    raise ValueError("Truncated baseline member")
        if set(contents) != set(limits):
            raise ValueError("Missing baseline outputs")
        metadata = json.loads(contents["metadata.json"])
        outputs = {
            case: {name: contents[f"outputs/{case}/{name}.bin"] for name in tensors}
            for case, tensors in contract.cases.items()
        }
        expected = _metadata(
            spec_sha256, suite, environment, metadata["image_id"], contract, outputs
        )
        if contents["metadata.json"] != expected:
            raise ValueError("Baseline provenance or descriptors mismatch")
        return BaselineBundleV2(metadata, outputs)
    except (tarfile.TarError, KeyError, TypeError, AttributeError, OverflowError, RecursionError):
        raise ValueError("Invalid cached baseline package") from None
