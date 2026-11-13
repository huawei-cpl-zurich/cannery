"""Publication controls and verified packages for the existing offline SDK."""

import json
import os
import re
import sys
from pathlib import Path
from time import monotonic, sleep
from typing import Annotated, Any
from uuid import UUID

import httpx
import typer

from cannery.devices import matches_soc
from cannery.git_intake.process import run
from cannery.releases.client import ReleaseClient
from cannery.releases.resolution import ResolveRequest

URL = "http://127.0.0.1:8000"
CACHE = Path.home() / ".cache" / "cannery" / "releases"


def emit(value: Any) -> None:
    typer.echo(json.dumps(value, sort_keys=True, indent=2))


def report_http_error(error: httpx.HTTPStatusError) -> None:
    try:
        detail = error.response.json()["detail"]
        code = detail.get("code") if isinstance(detail, dict) else detail
        if not isinstance(code, str) or re.fullmatch(r"[a-z][a-z0-9_]{0,79}", code) is None:
            raise ValueError("Unknown response")
        typer.echo(code, err=True)
        if isinstance(detail, dict) and isinstance(detail.get("cohorts"), list):
            for cohort in detail["cohorts"][:50]:
                if isinstance(cohort, str) and re.fullmatch(r"[a-f0-9]{64}", cohort):
                    typer.echo(f"--cohort {cohort}", err=True)
    except (ValueError, KeyError, TypeError):
        typer.echo(f"Request unavailable or invalid (HTTP {error.response.status_code})", err=True)


def action(identifier: UUID, kind: str, url: str, wait: bool, timeout: float) -> None:
    try:
        client = ReleaseClient(url)
        result = client.json(f"/api/v1/submissions/{identifier}/{kind}", {})
        emit(result)
        if kind != "publish" or not wait:
            return
        key = UUID(result["id"])
        deadline = monotonic() + timeout
        while monotonic() < deadline:
            result = client.json(f"/api/v1/publication-jobs/{key}", deadline=deadline)
            if result["status"] == "complete":
                emit(
                    client.json(f"/api/v1/submissions/{identifier}/publication", deadline=deadline)
                )
                return
            if result["status"] in {"failed", "cancelled"}:
                emit(result)
                raise typer.Exit(1)
            sleep(min(0.5, max(0, deadline - monotonic())))
        typer.echo(f"Still pending; resume with publication-status {identifier}", err=True)
        raise typer.Exit(1)
    except httpx.HTTPStatusError as error:
        report_http_error(error)
        raise typer.Exit(1) from None
    except (httpx.HTTPError, TimeoutError, ValueError, KeyError, TypeError):
        typer.echo(
            "Publication request unavailable or invalid; inspect its status before retrying",
            err=True,
        )
        raise typer.Exit(1) from None


def detect_runtime(timeout: float = 30) -> Any:
    return json.loads(
        run(
            [
                sys.executable,
                "-c",
                "import json; from cannery_kernels import runtime_fingerprint; "
                "print(json.dumps(runtime_fingerprint()))",
            ],
            env=dict(os.environ),
            timeout=timeout,
            max_output=16384,
            git_limits=False,
        )
    )


def install_release_cli(app: typer.Typer) -> None:
    @app.command()
    def publish(
        submission_id: UUID,
        url: str = URL,
        wait: bool = False,
        timeout: Annotated[float, typer.Option(min=1, max=900)] = 180,
    ) -> None:
        """Explicitly publish a completed submission; slower valid kernels are eligible."""
        action(submission_id, "publish", url, wait, timeout)

    @app.command()
    def withdraw(submission_id: UUID, url: str = URL) -> None:
        """Withdraw discovery and new downloads; retained offline bytes remain usable."""
        action(submission_id, "withdraw", url, False, 1)

    @app.command("publication-status")
    def status(submission_id: UUID, url: str = URL) -> None:
        """Observe an existing publication without creating a new request."""
        show(f"/api/v1/submissions/{submission_id}/publication", url)

    @app.command()
    def info(
        kernel: str,
        url: str = URL,
        device: str = "a5",
        spec_version: int | None = None,
        cohort: str | None = None,
        rtol: float | None = None,
        atol: float | None = None,
        limit: Annotated[int, typer.Option(min=1, max=100)] = 50,
        offset: Annotated[int, typer.Option(min=0)] = 0,
    ) -> None:
        """List published releases, saved tolerances, case latencies and immutable pins."""
        params = dict(
            kernel=kernel,
            device=device,
            spec_version=spec_version,
            cohort=cohort,
            rtol=rtol,
            atol=atol,
            limit=limit,
            offset=offset,
        )
        show("/api/v1/releases", url, {k: v for k, v in params.items() if v is not None})

    @app.command()
    def source(release_id: UUID, url: str = URL) -> None:
        """Print the pinned Git source; no submitted source code is executed."""
        show(f"/api/v1/releases/{release_id}", url, source_only=True)

    @app.command()
    def get(
        kernel: str,
        url: str = URL,
        device: str | None = None,
        release_id: UUID | None = None,
        language: str | None = None,
        spec_version: int | None = None,
        cohort: str | None = None,
        shape: str | None = None,
        rtol: float | None = None,
        atol: float | None = None,
        no_detect: bool = False,
        runtime_json: Path | None = None,
        cache: Path = CACHE,
    ) -> None:
        """Resolve and cache exact package bytes; never load native code."""
        try:
            if no_detect:
                if runtime_json is None or runtime_json.stat().st_size > 16384:
                    raise ValueError("--no-detect requires a small --runtime-json fingerprint")
                runtime = json.loads(runtime_json.read_bytes())
            else:
                if runtime_json is not None:
                    raise ValueError("Use --no-detect with --runtime-json")
                runtime = detect_runtime()
            selection = ResolveRequest.model_validate(
                {
                    "kernel": kernel,
                    "device": device
                    or ("a3" if matches_soc("a3", runtime.get("soc", "")) else "a5"),
                    "runtime": runtime,
                    "release_id": release_id,
                    "language": language,
                    "spec_version": spec_version,
                    "cohort": cohort,
                    "shape": None if shape is None else [int(v) for v in shape.split(",")],
                    "rtol": rtol,
                    "atol": atol,
                }
            )
            result = ReleaseClient(url).get(selection, cache)
            emit(
                {
                    "manifest": str(result.manifest_path),
                    "manifest_sha256": result.manifest_sha256,
                    "package": str(result.package_path),
                    "package_sha256": result.manifest.package.sha256,
                }
            )
        except httpx.HTTPStatusError as error:
            report_http_error(error)
            raise typer.Exit(1) from None
        except (
            httpx.HTTPError,
            TimeoutError,
            ValueError,
            KeyError,
            TypeError,
            OSError,
            ImportError,
            RuntimeError,
        ):
            typer.echo(
                "Download unavailable or incompatible; supply a matching runtime fingerprint",
                err=True,
            )
            raise typer.Exit(1) from None


def show(
    path: str, url: str, params: dict[str, Any] | None = None, *, source_only: bool = False
) -> None:
    try:
        result = ReleaseClient(url).json(path, params=params)
        emit(
            {"source": result["source"], "source_sha256": result["source_sha256"]}
            if source_only
            else result
        )
    except httpx.HTTPStatusError as error:
        report_http_error(error)
        raise typer.Exit(1) from None
    except (httpx.HTTPError, TimeoutError, ValueError, KeyError, TypeError):
        typer.echo("Published metadata unavailable or selection requires refinement", err=True)
        raise typer.Exit(1) from None
