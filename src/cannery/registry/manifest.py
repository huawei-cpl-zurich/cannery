"""Validate source bytes without importing author modules."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from cannery.git_intake import relative_source_path

Name = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")]


class SpecError(ValueError):
    pass


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class KernelIdentity(StrictModel):
    namespace: Name
    name: Name
    description: str = Field(default="", max_length=4096)


class Entrypoint(StrictModel):
    file: str
    function: Name

    @field_validator("file")
    @classmethod
    def safe_file(cls, value: str) -> str:
        return relative_source_path(value)


class Interface(StrictModel):
    inputs: tuple[Name, ...] = Field(min_length=1, max_length=32)
    outputs: tuple[Name, ...] = Field(min_length=1, max_length=32)

    @field_validator("inputs", "outputs", mode="before")
    @classmethod
    def names(cls, value: Any) -> Any:
        if isinstance(value, list):
            if len(set(value)) != len(value):
                raise ValueError("Duplicate interface name")
            return tuple(value)
        return value


class Validation(StrictModel):
    metric: Literal["torch_close"]
    equal_nan: bool = False
    default_rtol: float = Field(ge=0, allow_inf_nan=False)
    default_atol: float = Field(ge=0, allow_inf_nan=False)


class SpecDefinition(StrictModel):
    version: int = Field(ge=1)
    reference: Entrypoint
    cases: Entrypoint
    interface: Interface
    validation: Validation
    environment_lock: str = "environment.lock"

    @field_validator("environment_lock")
    @classmethod
    def safe_lock(cls, value: str) -> str:
        return relative_source_path(value)


class Manifest(StrictModel):
    schema_version: int = Field(ge=1, le=1)
    kernel: KernelIdentity
    spec: SpecDefinition


class EnvironmentLock(StrictModel):
    schema_version: int = Field(ge=1, le=1)
    python: str = Field(pattern=r"^3\.[0-9]+$")
    torch: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+(?:\+cpu)?$")


def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise SpecError("Duplicate mapping key")
        result[key] = value
    return result


def parse_manifest(data: bytes) -> Manifest:
    if len(data) > 65536:
        raise SpecError("Manifest exceeds 64 KiB")
    try:
        # Disallow aliases/anchors before constructing a graph; reject duplicate
        # keys, custom tags, and ambiguous YAML structures rather than coercing.
        for event in yaml.parse(data):
            if isinstance(event, yaml.AliasEvent) or getattr(event, "anchor", None):
                raise SpecError("YAML aliases and anchors are unsupported")
        node = yaml.compose(data)
        if not isinstance(node, yaml.MappingNode):
            raise SpecError("Manifest must be a mapping")

        def keys(item: yaml.Node) -> None:
            if isinstance(item, yaml.MappingNode):
                names = [key.value for key, _ in item.value]
                if len(set(names)) != len(names):
                    raise SpecError("Duplicate mapping key")
                for _, child in item.value:
                    keys(child)
            elif isinstance(item, yaml.SequenceNode):
                for child in item.value:
                    keys(child)

        keys(node)
        return Manifest.model_validate(yaml.safe_load(data))
    except (yaml.YAMLError, ValueError, TypeError, RecursionError):
        raise SpecError("Invalid specification manifest") from None


@dataclass(frozen=True)
class ValidatedSpec:
    manifest: Manifest
    environment: EnvironmentLock
    content_sha256: str
    file_sha256: dict[str, str]


def validate_spec(files: dict[str, bytes], source_path: str) -> ValidatedSpec:
    prefix = "" if source_path == "." else relative_source_path(source_path) + "/"
    try:
        manifest_path = prefix + "cannery.yaml"
        manifest = parse_manifest(files[manifest_path])
        required = [
            manifest.spec.reference.file,
            manifest.spec.cases.file,
            manifest.spec.environment_lock,
        ]
        for name in required:
            if prefix + name not in files:
                raise SpecError("Referenced spec file is missing")
        environment = EnvironmentLock.model_validate(
            json.loads(
                files[prefix + manifest.spec.environment_lock],
                object_pairs_hook=unique_pairs,
            )
        )
        for path in files:
            relative_source_path(path)
            if ".git" in PurePosixPath(path).parts:
                raise SpecError("Git administration paths are forbidden")
        hashes = {name: hashlib.sha256(data).hexdigest() for name, data in sorted(files.items())}
        normalized = manifest.model_dump(mode="json")
        normalized["kernel"].pop("description")
        normalized["spec"].pop("version")
        identity = {
            "manifest": normalized,
            "files": {name: sha for name, sha in hashes.items() if name != manifest_path},
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return ValidatedSpec(manifest, environment, digest, hashes)
    except (KeyError, ValueError, TypeError, RecursionError):
        raise SpecError("Invalid or incomplete specification files") from None
