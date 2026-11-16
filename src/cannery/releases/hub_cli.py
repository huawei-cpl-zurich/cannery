"""Explicit operator backfill; never rebuild or republish a kernel."""

import json

import typer
from sqlalchemy.exc import SQLAlchemyError

from cannery.database import make_engine
from cannery.releases.hub import backfill
from cannery.settings import Settings

app = typer.Typer(help="Private Hub snapshot operations.")


@app.command("backfill")
def snapshots() -> None:
    engine = make_engine(Settings())
    try:
        typer.echo(json.dumps({"created": backfill(engine)}))
    except (ValueError, SQLAlchemyError):
        typer.echo(
            "Hub backfill failed; retry after checking database and release integrity.", err=True
        )
        raise typer.Exit(1) from None
    finally:
        engine.dispose()
