"""Explicit offline operator actions, never kernel execution or silent service stops."""

import json
import tarfile
from pathlib import Path
from typing import Annotated

import typer

from cannery.maintenance.archive import verify
from cannery.maintenance.artifacts import audit
from cannery.maintenance.capture import Docker, capture
from cannery.maintenance.restore import restore

app = typer.Typer(
    help="Offline backups and empty-target restore; stop all application writers first."
)


def report(error: Exception) -> None:
    typer.echo(
        f"Maintenance failed: {error}. Keep restore targets stopped after a failure.", err=True
    )
    raise typer.Exit(1)


@app.command("backup")
def backup(
    project: Annotated[str, typer.Option()], output: Annotated[Path, typer.Option()]
) -> None:
    """Back up a stopped Compose project; PostgreSQL and Redis may remain running."""
    try:
        manifest = capture(Docker(), project, output)
        typer.echo(
            json.dumps(
                {"id": str(manifest.id), "backup": str(output), "files": len(manifest.files)}
            )
        )
    except (OSError, ValueError, tarfile.TarError) as error:
        report(error)


@app.command("verify-backup")
def check(directory: Path) -> None:
    """Verify hashes, exact file set and bounded safe archives without restoring."""
    try:
        manifest = verify(directory)
        typer.echo(
            json.dumps({"id": str(manifest.id), "project": manifest.project, "verified": True})
        )
    except (OSError, ValueError, tarfile.TarError) as error:
        report(error)


@app.command("restore")
def recover(directory: Path, project: Annotated[str, typer.Option()]) -> None:
    """Restore into a different empty stopped project; never start services."""
    try:
        manifest = restore(Docker(), directory, project)
        typer.echo(
            json.dumps(
                {
                    "id": str(manifest.id),
                    "project": project,
                    "restored": True,
                    "services_started": False,
                }
            )
        )
    except (OSError, ValueError, tarfile.TarError) as error:
        report(error)


@app.command("audit-artifacts")
def artifact_audit(
    project: Annotated[str, typer.Option()], age_hours: Annotated[float, typer.Option(min=24)] = 168
) -> None:
    """Read-only offline audit of retained references, missing bytes and aged orphans."""
    try:
        result = audit(Docker(), project, age_hours)
        typer.echo(json.dumps(result, sort_keys=True, indent=2))
        if result["errors"]:
            raise typer.Exit(1)
    except (OSError, ValueError, tarfile.TarError) as error:
        report(error)
