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
