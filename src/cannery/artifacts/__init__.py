"""Generated binary/evidence storage; never source ingestion or execution."""

import hashlib
from pathlib import PurePosixPath
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ArtifactError(Exception):
    """Artifact missing, unavailable, unsafe, or corrupt; no credentials included."""


class ArtifactRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    repository: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*(/[A-Za-z0-9][A-Za-z0-9_.-]*)?$")
    revision: str = Field(pattern=r"^(?:[a-f0-9]{40}|[a-f0-9]{64})$")
    path: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @field_validator("path")
    @classmethod
    def confined_path(cls, value: str) -> str:
        parts = value.split("/")
        if (
            not value
            or PurePosixPath(value).is_absolute()
            or "\\" in value
            or any(part in ("", ".", "..") for part in parts)
            or any(ord(char) < 32 for char in value)
        ):
            raise ValueError("Expected a confined relative artifact path")
        return value


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verified(data: bytes, reference: ArtifactRef) -> bytes:
    if digest(data) != reference.sha256:
        raise ArtifactError("Artifact checksum mismatch")
    return data


class ArtifactStore(Protocol):
    def put(self, path: str, data: bytes) -> ArtifactRef: ...
    def get(self, reference: ArtifactRef) -> bytes: ...
