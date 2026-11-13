"""Offline Docker-project inventory and atomic backup capture; no credentials exported."""

import ctypes
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO
from uuid import uuid4

from cannery.maintenance.archive import (
    PROJECT,
    ROLE_FILES,
    BackupFile,
    BackupManifest,
    Role,
    checksum,
    verify,
)

MOUNTS: dict[Role, tuple[str, str, str]] = {
    "snapshots": ("api", "CANNERY_SNAPSHOT_ROOT", "/var/lib/cannery/snapshots"),
    "artifacts": ("artifacts", "CANNERY_ARTIFACT_ROOT", "/home/cannery/artifacts"),
    "build_control": ("builder", "CANNERY_BUILD_CONTROL_ROOT", "/var/lib/cannery/builds"),
    "baseline_control": ("registrar", "CANNERY_BASELINE_ROOT", "/var/lib/cannery/baselines"),
}


class Docker:
    def run(
        self,
        *args: str,
        source: BinaryIO | None = None,
        destination: BinaryIO | None = None,
        timeout: int = 300,
    ) -> bytes:
        with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
            try:
                subprocess.run(
                    ["docker", *args],
                    stdin=source,
                    stdout=destination or output,
                    stderr=errors,
                    check=True,
                    timeout=timeout,
                )
            except (OSError, subprocess.SubprocessError):
                raise ValueError(
                    "Docker maintenance operation failed or exceeded deadline"
                ) from None
            if destination is not None:
                return b""
            if output.tell() > 4 * 1024 * 1024:
                raise ValueError("Docker response exceeds inventory budget")
            output.seek(0)
            return output.read()

    def inspect(self, *ids: str) -> list[dict[str, Any]]:
        return json.loads(self.run("inspect", *ids))  # type: ignore[no-any-return]

    def volume(
        self,
        image: str,
        name: str,
        program: str,
        *,
        destination: BinaryIO | None = None,
        source: BinaryIO | None = None,
        writable: bool = False,
    ) -> bytes:
        helper = "cannery-maintenance-" + uuid4().hex
        mount = f"type=volume,source={name},target=/data" + ("" if writable else ",readonly")
        try:
            return self.run(
                "run",
                "--rm",
                "--name",
                helper,
                "--label",
                "cannery.maintenance=true",
                "--network",
                "none",
                "--read-only",
                "--user",
                "0:0",
                "--cap-drop",
                "ALL",
                "--cap-add",
                "DAC_OVERRIDE",
                *(["--cap-add", "CHOWN", "--cap-add", "FOWNER"] if writable else []),
                "--security-opt",
                "no-new-privileges",
                "--pids-limit",
                "64",
                "--memory",
                "256m",
                "--mount",
                mount,
                "-i",
                image,
                "python",
                "-c",
                program,
                source=source,
                destination=destination,
            )
        finally:
            # A disconnected Docker client can leave its helper alive; stop only this unique name.
            try:
                if self.run("ps", "-aq", "--filter", f"name=^/{helper}$", timeout=15).strip():
                    self.run("stop", "-t", "1", helper, timeout=15)
                    if self.run("ps", "-aq", "--filter", f"name=^/{helper}$", timeout=15).strip():
                        self.run("rm", helper, timeout=15)
            except ValueError:
                raise ValueError(f"Maintenance helper cleanup pending: {helper}") from None


@dataclass
class Project:
    name: str
    postgres: str
    user: str
    database: str
    helper_image: str
    images: dict[str, str]
    volumes: dict[Role, str]
    docker: Docker

    def sql(self, query: str) -> str:
        return (
            self.docker.run(
                "exec",
                self.postgres,
                "psql",
                "-X",
                "-v",
                "ON_ERROR_STOP=1",
                "-U",
                self.user,
                "-d",
                self.database,
                "-Atc",
                query,
            )
            .decode()
            .strip()
        )


def environment(container: dict[str, Any]) -> dict[str, str]:
    return dict(value.split("=", 1) for value in container["Config"]["Env"] if "=" in value)


def inventory(docker: Docker, name: str) -> Project:
    if re.fullmatch(PROJECT, name) is None:
        raise ValueError("Invalid Compose project name")
    ids = (
        docker.run("ps", "-aq", "--filter", f"label=com.docker.compose.project={name}")
        .decode()
        .split()
    )
    if not ids:
        raise ValueError("Compose project not found")
    containers = docker.inspect(*ids)
    by_service: dict[str, dict[str, Any]] = {}
    images = {}
    for container in containers:
        labels = container["Config"]["Labels"]
        if labels.get("com.docker.compose.project") != name:
            raise ValueError("Project identity changed")
        service = labels["com.docker.compose.service"]
        if (
            service not in {"postgres", "redis"}
            and container["State"]["Status"] not in {"exited", "created"}
        ) or (container["State"]["Running"] and labels.get("com.docker.compose.oneoff") == "True"):
            raise ValueError("Stop all application writers before maintenance")
        if labels.get("com.docker.compose.oneoff") == "True":
            continue
        if service in by_service:
            raise ValueError("Maintenance requires one container per service")
        by_service[service] = container
        images[service] = container["Image"]
    if not {"postgres", "api", "artifacts"} <= by_service.keys():
        raise ValueError("Expected PostgreSQL, API and artifact services")
    postgres = by_service["postgres"]
    if not postgres["State"]["Running"]:
        raise ValueError("PostgreSQL must be running for maintenance")
    volumes: dict[Role, str] = {}
    for role, (service, variable, default) in MOUNTS.items():
        if service not in by_service:
            continue
        container = by_service[service]
        path = environment(container).get(variable, default)
        mount = next((m for m in container["Mounts"] if m["Destination"] == path), None)
        if mount is None or mount["Type"] != "volume":
            if role == "baseline_control" and variable not in environment(container):
                continue  # Source-intake-only deployments have no CPU baseline volume.
            raise ValueError(f"Expected named volume for {role}")
        volumes[role] = mount["Name"]
    if len(set(volumes.values())) != len(volumes):
        raise ValueError("Maintenance volumes must be distinct")
    # Outside readers are harmless; an outside writer would break snapshot consistency.
    for volume in volumes.values():
        users = docker.run("ps", "-q", "--filter", f"volume={volume}").decode().split()
        if users and any(
            m.get("Name") == volume and m.get("RW")
            for c in docker.inspect(*users)
            for m in c["Mounts"]
        ):
            raise ValueError("A running container can write a maintenance volume")
    env = environment(postgres)
    user = env.get("POSTGRES_USER", "postgres")
    return Project(
        name,
        postgres["Id"],
        user,
        env.get("POSTGRES_DB", user),
        by_service["api"]["Image"],
        images,
        volumes,
        docker,
    )


TAR = "import os; os.execvp('tar', ['tar','-C','/data','-cf','-','.'])"


def publish_directory(source: Path, destination: Path) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    rename = getattr(library, "renameat2", None)
    if rename is None:
        raise ValueError("Atomic no-replace directory rename is unavailable")
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1):
        raise OSError(ctypes.get_errno(), "Cannot publish backup without replacing destination")


def capture(docker: Docker, project: str, destination: Path) -> BackupManifest:
    current = inventory(docker, project)
    if destination.exists() or destination.is_symlink():
        raise ValueError("Backup destination must not exist")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".cannery-backup-", dir=destination.parent))
    try:
        files = []
        database = temporary / ROLE_FILES["database"]
        with database.open("xb") as stream:
            os.chmod(database, 0o600)
            docker.run(
                "exec",
                current.postgres,
                "pg_dump",
                "-U",
                current.user,
                "-d",
                current.database,
                "-Fc",
                destination=stream,
            )
            stream.flush()
            os.fsync(stream.fileno())
        for role, volume in current.volumes.items():
            path = temporary / ROLE_FILES[role]
            with path.open("xb") as stream:
                os.chmod(path, 0o600)
                docker.volume(current.helper_image, volume, TAR, destination=stream)
                stream.flush()
                os.fsync(stream.fileno())
        for role in ("database", *current.volumes):
            size, sha = checksum(temporary / ROLE_FILES[role])
            files.append(BackupFile(role=role, size=size, sha256=sha))
        manifest = BackupManifest(
            id=uuid4(),
            created_at=datetime.now(UTC),
            project=project,
            postgres_version=int(current.sql("SHOW server_version_num")),
            migration=current.sql("SELECT version_num FROM alembic_version"),
            images=current.images,
            files=tuple(files),
        )
        path = temporary / "manifest.json"
        with path.open("xb") as stream:
            os.chmod(path, 0o600)
            stream.write(manifest.encoded())
            stream.flush()
            os.fsync(stream.fileno())
        verify(temporary)
        if inventory(docker, project) != current:
            raise ValueError("Project changed during backup")
        publish_directory(temporary, destination)
        descriptor = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
