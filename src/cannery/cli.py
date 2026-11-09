import typer

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    """CANNery control-plane commands."""


@app.command()
def version() -> None:
    """Print the installed CANNery version."""
    from importlib.metadata import version as installed_version

    typer.echo(installed_version("cannery"))


@app.command()
def health(url: str = "http://127.0.0.1:8000") -> None:
    """Check control-plane readiness; exit nonzero if unavailable."""
    import httpx

    try:
        response = httpx.get(f"{url.rstrip('/')}/health/ready", timeout=5)
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict) or body.get("status") != "ok":
            raise ValueError("Invalid health response")
    except (httpx.HTTPError, ValueError):
        typer.echo("CANNery is unavailable", err=True)
        raise typer.Exit(1) from None
    typer.echo("CANNery is ready")
