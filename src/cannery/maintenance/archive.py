"""Strict, bounded backup verification before any restore side effects."""

import hashlib
import json
import os
import re
import stat
import tarfile
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Literal, Self
from uuid import UUID

from pydantic import Field, model_validator

from cannery.registry.manifest import StrictModel, unique_pairs

SHA = r"^[a-f0-9]{64}$"
PROJECT = r"^[a-z0-9][a-z0-9_-]{0,62}$"
ROLE_FILES = {
    "database": "database.dump",
    "snapshots": "git-snapshots.tar",
    "artifacts": "artifact-data.tar",
    "build_control": "build-control.tar",
    "baseline_control": "baseline-control.tar",
}
Role = Literal["database", "snapshots", "artifacts", "build_control", "baseline_control"]
MAX_BYTES = 1024**4


class BackupFile(StrictModel):
    role: Role
    size: int = Field(ge=1, le=MAX_BYTES)
    sha256: str = Field(pattern=SHA)

    @property
    def filename(self) -> str:
        return ROLE_FILES[self.role]


class BackupManifest(StrictModel):
    schema_version: Literal[1] = 1
    id: UUID
    created_at: datetime
    project: str = Field(pattern=PROJECT)
    postgres_version: int = Field(ge=170000, lt=180000)
    migration: str = Field(pattern=r"^[0-9]{4}$")
    images: dict[str, str] = Field(min_length=1, max_length=32)
    files: tuple[BackupFile, ...] = Field(min_length=3, max_length=5)

    @model_validator(mode="after")
    def coherent(self) -> Self:
        roles = [f.role for f in self.files]
        if len(set(roles)) != len(roles) or not {"database", "snapshots", "artifacts"} <= set(
            roles
        ):
            raise ValueError("Backup requires unique database, snapshot and artifact files")
        if sum(f.size for f in self.files) > MAX_BYTES:
            raise ValueError("Backup exceeds total byte budget")
        if self.created_at.utcoffset() is None:
            raise ValueError("Backup timestamp must include timezone")
        if any(
            not re.fullmatch(PROJECT, k) or not re.fullmatch("sha256:" + SHA[1:], v)
            for k, v in self.images.items()
        ):
            raise ValueError("Backup images must be immutable image IDs")
        return self

    def encoded(self) -> bytes:
        return json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()


def open_regular(path: Path) -> BinaryIO:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ValueError("Backup files must be regular files")
    return os.fdopen(descriptor, "rb")


def checksum(path: Path) -> tuple[int, str]:
    size, sha = 0, hashlib.sha256()
    with open_regular(path) as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_BYTES:
                raise ValueError("Backup exceeds byte budget")
            sha.update(chunk)
    return size, sha.hexdigest()


def pax_metadata(data: bytes) -> None:
    while data:
        space = data.find(b" ")
        if not 0 < space <= 8 or not data[:space].isdigit():
            raise ValueError("Invalid PAX record")
        length = int(data[:space])
        if not space + 3 <= length <= len(data) or data[length - 1 : length] != b"\n":
            raise ValueError("Invalid PAX record size")
        key = data[space + 1 : length - 1].split(b"=", 1)[0]
        if key not in {b"path", b"mtime", b"atime", b"ctime", b"uname", b"gname", b"comment"}:
            raise ValueError("Unsupported PAX key (size and sparse overrides are forbidden)")
        data = data[length:]


def preflight(stream: BinaryIO) -> int:
    size = stream.seek(0, os.SEEK_END)
    stream.seek(0)
    extensions = 0
    for _ in range(1000000):
        header = stream.read(512)
        if len(header) != 512:
            raise ValueError("Truncated tar header")
        if header == bytes(512):
            if stream.read(512) != bytes(512):
                raise ValueError("Missing tar end marker")
            while padding := stream.read(65536):
                if any(padding):
                    raise ValueError("Unexpected data after tar end marker")
            stream.seek(0)
            return size
        item = tarfile.TarInfo.frombuf(header, "utf-8", "surrogateescape")
        if item.isdir() and item.size != 0:
            raise ValueError("Directory payloads are unsupported")
        padded = (item.size + 511) // 512 * 512
        if item.size < 0 or padded > size - stream.tell():
            raise ValueError("Truncated tar member")
        if item.type in (tarfile.XHDTYPE, tarfile.GNUTYPE_LONGNAME):
            extensions += 1
            if item.size > 16384 or extensions > 16:
                raise ValueError("Tar extension metadata exceeds budget")
            data = stream.read(item.size)
            if b"GNU.sparse" in data:
                raise ValueError("Sparse metadata is unsupported")
            if item.type == tarfile.XHDTYPE:
                pax_metadata(data)
            stream.seek(padded - item.size, os.SEEK_CUR)
        elif item.type in (tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE):
            extensions = 0
            stream.seek(padded, os.SEEK_CUR)
        else:
            raise ValueError("Unsafe backup archive member")
    raise ValueError("Tar header count exceeds budget")


def members(stream: BinaryIO) -> Iterator[tarfile.TarInfo]:
    archive_size = preflight(stream)
    seen: set[str] = set()
    paths: dict[str, bool] = {}
    path_bytes = 0
    total = 0
    with tarfile.open(fileobj=stream, mode="r|", errorlevel=2) as archive:
        for item in archive:
            path = PurePosixPath(item.name)
            name = str(path)
            total += item.size
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in name
                or "\0" in name
                or name == "."
                and not item.isdir()
                or name in seen
                or not (item.isfile() or item.isdir())
                or item.mode & 0o7000
                or item.size < 0
                or total > MAX_BYTES
                or len(seen) >= 1000000
                or len(name.encode()) > 4096
                or item.sparse is not None
                or item.offset_data + item.size > archive_size
            ):
                raise ValueError("Unsafe backup archive member")
            parents = [str(p) for p in path.parents]
            if any(paths.get(p) is False for p in parents) or (item.isfile() and name in paths):
                raise ValueError("Conflicting backup archive paths")
            for parent in parents:
                if parent not in paths:
                    path_bytes += len(parent.encode())
                    paths[parent] = True
            if name not in paths:
                path_bytes += len(name.encode())
            paths[name] = item.isdir()
            if path_bytes > 64 * 1024 * 1024 or len(paths) > 1000000:
                raise ValueError("Archive path inventory exceeds budget")
            seen.add(name)
            # Python 3.12 otherwise retains every parsed TarInfo, including PAX metadata.
            archive.members.clear()  # type: ignore[attr-defined]
            yield item


def verify(directory: Path) -> BackupManifest:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Backup must be a directory")
    with open_regular(directory / "manifest.json") as stream:
        raw = stream.read(128 * 1024 + 1)
    if len(raw) > 128 * 1024:
        raise ValueError("Backup manifest exceeds budget")
    document = json.loads(raw, object_pairs_hook=unique_pairs)
    manifest = BackupManifest.model_validate_json(json.dumps(document))
    if manifest.encoded() != raw:
        raise ValueError("Backup manifest is not canonical")
    if {p.name for p in directory.iterdir()} != {"manifest.json"} | {
        f.filename for f in manifest.files
    }:
        raise ValueError("Backup file set differs from manifest")
    for file in manifest.files:
        path = directory / file.filename
        if checksum(path) != (file.size, file.sha256):
            raise ValueError("Backup file checksum or size mismatch")
        with open_regular(path) as stream:
            if file.role == "database":
                if stream.read(5) != b"PGDMP":
                    raise ValueError("Expected a PostgreSQL custom-format dump")
            else:
                for _ in members(stream):
                    pass
    return manifest
