"""Git-only intake contracts. External fetching/importing is implemented in M1."""

from typing import Literal, Protocol
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator


def relative_source_path(value: str) -> str:
    if (
        not value
        or value.startswith("/")
        or "\\" in value
        or any(ord(char) < 32 for char in value)
        or any(part in ("", ".", "..") for part in value.split("/"))
    ):
        raise ValueError("Expected a confined repository-relative path")
    return value


class GitSource(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["git"] = "git"
    url: str
    commit: str = Field(pattern=r"^(?:[a-f0-9]{40}|[a-f0-9]{64})$")
    path: str
    dependencies: tuple[str, ...] = ()

    @field_validator("url")
    @classmethod
    def remote_url(cls, value: str) -> str:
        if any(char.isspace() or ord(char) < 32 for char in value):
            raise ValueError("Invalid Git remote URL")
        parsed = urlsplit(value)
        if (
            parsed.scheme not in ("https", "ssh")
            or not parsed.hostname
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or not parsed.path
            or parsed.path == "/"
            or (
                parsed.username is not None
                and not (parsed.scheme == "ssh" and parsed.username == "git")
            )
        ):
            raise ValueError("Expected an HTTPS or ssh://git@ Git remote without credentials")
        # Validate any supplied port rather than accepting a malformed URL.
        _ = parsed.port
        return value

    @field_validator("path")
    @classmethod
    def source_path(cls, value: str) -> str:
        value = value[:-1] if value.endswith("/") else value
        return value if value == "." else relative_source_path(value)

    @field_validator("dependencies")
    @classmethod
    def dependency_paths(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(relative_source_path(value) for value in values)


class SnapshotBinding(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source: GitSource
    internal_repository: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    internal_snapshot_sha: str = Field(pattern=r"^(?:[a-f0-9]{40}|[a-f0-9]{64})$")
    external_tree_id: str = Field(pattern=r"^(?:[a-f0-9]{40}|[a-f0-9]{64})$")
    content_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class SnapshotImporter(Protocol):
    def import_snapshot(self, kernel_id: str, source: GitSource) -> SnapshotBinding:
        """Retain selected files/dependencies and provenance, never the upstream DAG.

        M1 must enforce host/IP authorization, resource limits, tree/path safety,
        protected refs, and cleanup before implementing this method. This protocol
        alone is not a safe remote-fetch implementation.
        """
        ...
