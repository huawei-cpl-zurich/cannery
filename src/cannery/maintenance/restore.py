"""Restore verified bytes only into an empty, stopped, independently owned project."""

import json
from pathlib import Path

from cannery.maintenance.archive import BackupManifest, open_regular, verify
from cannery.maintenance.capture import Docker, inventory

EMPTY = "import os; assert not any(os.scandir('/data')), 'Restore volume is not empty'"
EXTRACT = "import os; os.execvp('tar', ['tar','-C','/data','--numeric-owner','-xpf','-'])"


def restore(docker: Docker, directory: Path, target: str) -> BackupManifest:
    manifest = verify(directory)
    if target == manifest.project:
        raise ValueError("Restore requires a different isolated target project")
    project = inventory(docker, target)
    if int(project.sql("SHOW server_version_num")) // 10000 != manifest.postgres_version // 10000:
        raise ValueError("Restore PostgreSQL major version differs")
    if (
        project.sql(
            "SELECT count(*) FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
            "WHERE n.nspname NOT IN ('pg_catalog','information_schema') "
            "AND n.nspname NOT LIKE 'pg_toast%'"
        )
        != "0"
    ):
        raise ValueError("Restore database must be empty")
    roles = {file.role for file in manifest.files if file.role != "database"}
    if roles != project.volumes.keys():
        raise ValueError("Restore target volume roles differ from backup")
    # Prevent accidental reuse of another project's externally mounted data volumes.
    for volume in project.volumes.values():
        details = docker.run("volume", "inspect", volume)
        labels = json.loads(details)[0].get("Labels") or {}
        if labels.get("com.docker.compose.project") != target:
            raise ValueError("Restore volume is not owned by the target project")
        docker.volume(project.helper_image, volume, EMPTY)
    # Repeat all guards after verification/empty checks, before the first write.
    if inventory(docker, target) != project:
        raise ValueError("Restore project changed")
    for file in manifest.files:
        if file.role != "database":
            with open_regular(directory / file.filename) as stream:
                docker.volume(
                    project.helper_image,
                    project.volumes[file.role],
                    EXTRACT,
                    source=stream,
                    writable=True,
                )
    with open_regular(directory / "database.dump") as stream:
        docker.run(
            "exec",
            "-i",
            project.postgres,
            "pg_restore",
            "-U",
            project.user,
            "-d",
            project.database,
            "--exit-on-error",
            "--single-transaction",
            "--no-owner",
            source=stream,
        )
    if project.sql("SELECT version_num FROM alembic_version") != manifest.migration:
        raise ValueError("Restored schema differs; keep target stopped")
    verify(directory)  # Detect source corruption during transfer; never start a failed target.
    return manifest  # No service start, migration, job dispatch or baseline execution.
