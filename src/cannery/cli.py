from typing import Annotated

import typer

from cannery.accounts.cli import app as accounts
from cannery.accounts.client import headers as auth_headers
from cannery.accounts.client_cli import install as install_account_cli
from cannery.maintenance.cli import app as maintenance
from cannery.releases.cli import install_release_cli
from cannery.submissions.cli import install_cli

app = typer.Typer(no_args_is_help=True)
install_account_cli(app)
app.add_typer(accounts, name="accounts")
app.add_typer(maintenance, name="maintenance")


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


@app.command()
def register(
    repo: Annotated[str, typer.Option()],
    commit: Annotated[str, typer.Option()],
    path: Annotated[str, typer.Option()],
    url: str = "http://127.0.0.1:8000",
    namespace: str | None = None,
    name: str | None = None,
    dependency: Annotated[list[str] | None, typer.Option("--dependency")] = None,
    wait: bool = False,
    timeout: Annotated[float, typer.Option(min=1, max=900)] = 180,
) -> None:
    """Register a specification from a pinned Git commit; optionally wait for its outcome."""
    import time

    import httpx

    from cannery.git_intake import GitSource

    try:
        source = GitSource(url=repo, commit=commit, path=path, dependencies=tuple(dependency or ()))
        body = {"source": source.model_dump(mode="json"), "namespace": namespace, "name": name}
        with httpx.Client(timeout=5, headers=auth_headers(url)) as client:
            response = client.post(url.rstrip("/") + "/api/v1/kernels", json=body)
            response.raise_for_status()
            key = response.json()["id"]
            typer.echo(f"Registration {key}")
            if not wait:
                return
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                try:
                    reply = client.get(url.rstrip("/") + f"/api/v1/registrations/{key}")
                    reply.raise_for_status()
                    result = reply.json()
                except httpx.HTTPStatusError as error:
                    if error.response.status_code < 500:
                        raise
                    time.sleep(0.5)
                    continue
                except httpx.TransportError:
                    time.sleep(0.5)
                    continue
                if result["status"] == "complete":
                    spec = result["spec"]
                    typer.echo(
                        f"Registered {spec['kernel']['namespace']}/{spec['kernel']['name']}"
                        f"@spec-v{spec['version']}"
                    )
                    return
                if result["status"] == "failed":
                    typer.echo(
                        "Registration failed; inspect its status for the error code", err=True
                    )
                    raise typer.Exit(1)
                time.sleep(0.5)
            typer.echo(
                "Still pending; inspect this registration ID to resume observation", err=True
            )
            raise typer.Exit(1)
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        typer.echo("Registration request unavailable or invalid", err=True)
        raise typer.Exit(1) from None


@app.command()
def registrations(url: str = "http://127.0.0.1:8000") -> None:
    """List recent owned registration IDs and outcomes."""
    import httpx

    try:
        response = httpx.get(
            url.rstrip("/") + "/api/v1/registrations", timeout=5, headers=auth_headers(url)
        )
        response.raise_for_status()
        for job in response.json():
            typer.echo(f"{job['id']} {job['status']}")
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        raise typer.Exit(1) from None


install_cli(app)
install_release_cli(app)
