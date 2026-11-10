"""Validate a source/build declaration without executing or compiling author code."""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal, Self

from pydantic import (
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

from cannery.git_intake import relative_source_path
from cannery.git_intake.snapshots import MAX_BYTES, MAX_FILES
from cannery.registry.manifest import Name, StrictModel, parse_yaml_mapping

KernelPath = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}/[a-z][a-z0-9_]{0,63}$")]
Language = Literal["ascendc", "catlass"]
Device = Literal["a3", "a5"]


class SubmissionError(ValueError):
    pass


class BuildOptions(StrictModel):
    optimization: Literal["release", "debug"] = "release"


class Build(StrictModel):
    adapter: Language
    abi: Literal["unary-f32-v1"] | None = None
    inputs: tuple[str, ...] = Field(min_length=1, max_length=128)
    artifact: Name
    options: BuildOptions = Field(default_factory=BuildOptions)

    @model_serializer(mode="wrap")
    def serialize(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        value: dict[str, Any] = handler(self)
        # Existing M2 hashes and frozen manifest representations must stay identical.
        if self.abi is None:
            value.pop("abi", None)
        return value

    @field_validator("inputs", mode="before")
    @classmethod
    def input_paths(cls, value: Any) -> tuple[str, ...]:
        if not isinstance(value, (list, tuple)) or not all(isinstance(v, str) for v in value):
            raise ValueError("Build inputs must be relative source paths")
        paths = tuple(relative_source_path(path) for path in value)
        if len(set(paths)) != len(paths):
            raise ValueError("Duplicate build input")
        return paths


class Implementation(StrictModel):
    device: Device
    language: Language
    entrypoint: str
    build: Build

    @model_validator(mode="after")
    def supported_contract(self) -> Self:
        if self.build.adapter != self.language:
            raise ValueError("Build adapter must match language")
        if self.build.abi is not None and (self.language != "ascendc" or self.device != "a5"):
            raise ValueError("unary-f32-v1 requires Ascend C on A5")
        parts = self.entrypoint.split(":")
        if len(parts) != 2 or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", parts[1]):
            raise ValueError("Entrypoint must be source-file:symbol")
        entry = relative_source_path(parts[0])
        extensions = (
            {".py"}
            if self.language == "catlass"
            else {".c", ".cc", ".cpp", ".cce", ".asc", ".h", ".hpp"}
        )
        if any(
            PurePosixPath(path).suffix not in extensions for path in (*self.build.inputs, entry)
        ):
            raise ValueError("Declared inputs must be supported source files, not binaries")
        if entry not in self.build.inputs:
            raise ValueError("Entrypoint source must be a declared build input")
        return self


class RequestedValidation(StrictModel):
    requested_rtol: float = Field(ge=0, allow_inf_nan=False)
    requested_atol: float = Field(ge=0, allow_inf_nan=False)


class SubmissionManifest(StrictModel):
    schema_version: int = Field(ge=1, le=1)
    kernel: KernelPath
    spec_version: int = Field(ge=1)
    implementation: Implementation
    validation: RequestedValidation


@dataclass(frozen=True)
class ValidatedSubmission:
    manifest: SubmissionManifest
    content_sha256: str
    file_sha256: dict[str, str]


def validate_submission(files: dict[str, bytes], source_path: str) -> ValidatedSubmission:
    try:
        prefix = "" if source_path == "." else relative_source_path(source_path) + "/"
        if len(files) > MAX_FILES or sum(map(len, files.values())) > MAX_BYTES:
            raise SubmissionError("Source exceeds intake budget")
        manifest_path = prefix + "submission.yaml"
        manifest = SubmissionManifest.model_validate(parse_yaml_mapping(files[manifest_path]))
        for path in files:
            relative_source_path(path)
            if ".git" in PurePosixPath(path).parts:
                raise SubmissionError("Git administration paths are forbidden")
        if any(prefix + path not in files for path in manifest.implementation.build.inputs):
            raise SubmissionError("Declared build input is missing")
        hashes = {name: hashlib.sha256(data).hexdigest() for name, data in sorted(files.items())}
        identity = {
            "manifest": manifest.model_dump(mode="json"),
            "files": {name: sha for name, sha in hashes.items() if name != manifest_path},
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return ValidatedSubmission(manifest, digest, hashes)
    except (KeyError, ValueError, TypeError, RecursionError):
        raise SubmissionError("Invalid or incomplete submission files") from None
