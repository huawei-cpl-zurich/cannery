"""Submission commands with separate source-intake and binary-admission waits."""

import json
from time import monotonic, sleep
from typing import Annotated, Any
from uuid import UUID

import httpx
import typer
from pydantic import TypeAdapter

from cannery.accounts.client import headers as auth_headers
from cannery.git_intake import GitSource
from cannery.submissions.manifest import Device, KernelPath, Language


def send(
    url: str,
    endpoint: str,
    body: dict[str, Any],
    wait: bool,
    timeout: float,
    *,
    wait_build: bool = False,
) -> None:
    try:
        with httpx.Client(timeout=5, headers=auth_headers(url), trust_env=False) as client:
            response = client.post(url.rstrip("/") + endpoint, json=body)
            response.raise_for_status()
            key = UUID(response.json()["id"])
            typer.echo(f"Submission {key}")
            if not wait:
                return
            deadline = monotonic() + timeout
            while monotonic() < deadline:
                try:
                    reply = client.get(url.rstrip("/") + f"/api/v1/submissions/{key}")
                    reply.raise_for_status()
                    result = reply.json()
                    if not isinstance(result, dict):
                        raise ValueError("Invalid status")
                except httpx.HTTPStatusError as error:
                    if error.response.status_code < 500:
                        raise
                    sleep(0.5)
                    continue
                except httpx.TransportError:
                    sleep(0.5)
                    continue
                if not wait_build and (
                    result["status"] == "awaiting_build"
                    or result.get("stages", {}).get("intake") == "complete"
                ):
                    typer.echo(
                        "Source retained; awaiting build. ETA unknown."
                        if result["status"] == "awaiting_build"
                        else "Source retained; intake complete. ETA unknown."
                    )
                    return
                if wait_build and (
                    result["status"] == "awaiting_validation"
                    or result.get("stages", {}).get("build") == "complete"
                ):
                    typer.echo(
                        "Binary admitted; awaiting validation. ETA unknown."
                        if result["status"] == "awaiting_validation"
                        else f"Binary admitted. Current status: {result['status']}. ETA unknown."
                    )
                    return
                if wait_build and result["status"] == "awaiting_build":
                    candidate = result.get("candidate")
                    if candidate and not candidate.get("manifest", {}).get(
                        "implementation", {}
                    ).get("build", {}).get("abi"):
                        typer.echo(
                            "Source retained; this source-only contract has no native build.",
                            err=True,
                        )
                        raise typer.Exit(1)
                if result["status"] == "failed":
                    typer.echo(f"Submission failed: {result.get('error', 'unknown')}", err=True)
                    raise typer.Exit(1)
                sleep(0.5)
            typer.echo("Still pending; inspect this submission ID to resume observation", err=True)
            raise typer.Exit(1)
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        typer.echo("Submission request unavailable or invalid", err=True)
        raise typer.Exit(1) from None


def install_cli(app: typer.Typer) -> None:
    @app.command()
    def submit(
        kernel: str,
        repo: Annotated[str, typer.Option()],
        commit: Annotated[str, typer.Option()],
        path: Annotated[str, typer.Option()],
        spec_version: Annotated[int, typer.Option(min=1)],
        device: Annotated[Device, typer.Option()],
        lang: Annotated[Language, typer.Option()],
        dependency: Annotated[list[str] | None, typer.Option("--dependency")] = None,
        url: str = "http://127.0.0.1:8000",
        wait: bool = False,
        wait_build: bool = False,
        timeout: Annotated[float, typer.Option(min=1, max=900)] = 180,
    ) -> None:
        """Submit Git source; --wait observes intake, --wait-build observes binary admission."""
        try:
            TypeAdapter(KernelPath).validate_python(kernel)
            namespace, name = kernel.split("/")
            source = GitSource(
                url=repo, commit=commit, path=path, dependencies=tuple(dependency or ())
            )
        except ValueError:
            typer.echo("Expected namespace/name and a valid pinned Git source", err=True)
            raise typer.Exit(1) from None
        send(
            url,
            f"/api/v1/kernels/{namespace}/{name}/submissions",
            {
                "spec_version": spec_version,
                "device": device,
                "language": lang,
                "source": source.model_dump(mode="json"),
            },
            wait or wait_build,
            timeout,
            wait_build=wait_build,
        )

    @app.command()
    def revise(
        identifier: UUID,
        commit: Annotated[str, typer.Option()],
        url: str = "http://127.0.0.1:8000",
        wait: bool = False,
        wait_build: bool = False,
        timeout: Annotated[float, typer.Option(min=1, max=900)] = 180,
    ) -> None:
        """Create an immutable revision inheriting the parent's frozen binding."""
        send(
            url,
            f"/api/v1/submissions/{identifier}/revisions",
            {"commit": commit},
            wait or wait_build,
            timeout,
            wait_build=wait_build,
        )

    @app.command()
    def submissions(url: str = "http://127.0.0.1:8000") -> None:
        """List recent owned submission IDs, status and ETA."""
        try:
            response = httpx.get(
                url.rstrip("/") + "/api/v1/submissions/mine", timeout=5, headers=auth_headers(url)
            )
            response.raise_for_status()
            for job in response.json():
                typer.echo(f"{job['id']} {job['status']} ETA unknown")
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            raise typer.Exit(1) from None

    @app.command()
    def submission(identifier: UUID, url: str = "http://127.0.0.1:8000") -> None:
        """Inspect owned source, stage status, numerical and private performance reports."""
        try:
            response = httpx.get(
                url.rstrip("/") + f"/api/v1/submissions/{identifier}",
                timeout=5,
                headers=auth_headers(url),
                trust_env=False,
            )
            response.raise_for_status()
            typer.echo(json.dumps(response.json(), indent=2))
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            raise typer.Exit(1) from None
