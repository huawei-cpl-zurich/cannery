"""Explicit deletion of aged orphans only after verifying recoverable backup copies."""

import hashlib
import inspect
import json
import os
import stat
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from cannery.maintenance.archive import open_regular, verify
from cannery.maintenance.artifacts import audit
from cannery.maintenance.capture import Docker, inventory


def unlink_verified(expected: list[dict[str, Any]]) -> None:
    root = Path("/data")
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for row in expected:
            path = root / row["name"]
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or (info.st_size, info.st_mtime_ns, info.st_ino) != (
                row["size"],
                row["mtime_ns"],
                row["inode"],
            ):
                raise ValueError("Artifact changed before deletion")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
            if digest.hexdigest() != row["sha256"]:
                raise ValueError("Artifact changed before deletion")
            path.unlink()
    finally:
        os.fsync(descriptor)
        os.close(descriptor)


def collect(docker: Docker, name: str, backup: Path, age_hours: float = 168) -> dict[str, Any]:
    manifest = verify(backup)
    if manifest.project != name:
        raise ValueError("Collection backup belongs to another project")
    result = audit(docker, name, age_hours)
    if result["errors"]:
        raise ValueError("Repair missing, corrupt or unknown artifacts before collection")
    expected = {row["name"]: row for row in result["candidates"]}
    copied = set()
    artifact = next(file for file in manifest.files if file.role == "artifacts")
    with (
        open_regular(backup / artifact.filename) as stream,
        tarfile.open(fileobj=stream, mode="r|") as archive,
    ):
        for member in archive:
            candidate = expected.get(str(PurePosixPath(member.name)))
            if candidate is not None and member.isfile():
                if member.size != candidate["size"]:
                    raise ValueError("Backup artifact size differs")
                content = archive.extractfile(member)
                if content is None:
                    raise ValueError("Backup artifact is missing")
                digest = hashlib.sha256()
                while chunk := content.read(1024 * 1024):
                    digest.update(chunk)
                if digest.hexdigest() != candidate["sha256"]:
                    raise ValueError("Backup artifact differs")
                copied.add(candidate["name"])
            archive.members.clear()  # type: ignore[attr-defined]
    if copied != expected.keys():
        raise ValueError("Collection requires a backup copy of every candidate")
    project = inventory(docker, name)
    if audit(docker, name, age_hours) != result:
        raise ValueError("Artifact audit changed; retry maintenance")
    program = (
        "from __future__ import annotations\nimport os,stat,hashlib,json,sys\n"
        "from pathlib import Path\n"
        "from typing import Any\n"
        + inspect.getsource(unlink_verified)
        + "\nunlink_verified(json.load(sys.stdin))"
    )
    with tempfile.TemporaryFile() as source:
        source.write(json.dumps(list(expected.values())).encode())
        source.seek(0)
        docker.volume(
            project.helper_image,
            project.volumes["artifacts"],
            program,
            source=source,
            writable=True,
        )
    return result | {"deleted": sorted(expected), "backup_id": str(manifest.id)}
