"""Authorized release bytes through the API; backend credentials stay internal."""

from typing import Annotated, Any, cast
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import Response
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from cannery.accounts.context import identity
from cannery.artifacts import ArtifactError, ArtifactRef, ArtifactStore
from cannery.releases import catalog
from cannery.releases.assets import verify_package
from cannery.releases.contract import read_manifest
from cannery.releases.discovery_api import selection_error
from cannery.releases.resolution import ResolveRequest, resolve
from cannery.settings import Settings


def install_downloads(app: FastAPI, settings: Settings, engine: Engine) -> None:
    if not settings.release_enabled:
        return

    def principal() -> str:
        return identity(settings)

    @app.post("/api/v1/artifacts/resolve")
    def resolution(
        body: ResolveRequest, user: Annotated[str, Depends(principal)]
    ) -> dict[str, Any]:
        try:
            with Session(engine) as db:
                return resolve(db, body)
        except catalog.SelectionError as error:
            raise selection_error(error) from None

    def download(identifier: UUID, package: bool) -> Response:
        try:
            with Session(engine) as db:
                release, _ = catalog.release_info(db, identifier)
                manifest = catalog.manifest_of(release)
                ref = manifest.package if package else ArtifactRef.model_validate(release.manifest)
            store = cast(ArtifactStore, app.state.artifacts)
            raw = store.get(ref)
            if package:
                verify_package(raw, manifest)
            elif read_manifest(raw, ref.sha256) != manifest:
                raise ValueError("Manifest differs")
            # Withdrawal during a slow backend read takes effect before any response bytes.
            with Session(engine) as db:
                catalog.release_info(db, identifier)
            return Response(
                raw,
                media_type="application/octet-stream" if package else "application/json",
                headers={"Cache-Control": "private, no-store", "ETag": '"' + ref.sha256 + '"'},
            )
        except catalog.SelectionError as error:
            raise selection_error(error) from None
        except (ArtifactError, ValueError):
            raise HTTPException(503, "release_artifact_unavailable") from None

    @app.get("/api/v1/releases/{identifier}/manifest")
    def manifest(identifier: UUID, user: Annotated[str, Depends(principal)]) -> Response:
        return download(identifier, False)

    @app.get("/api/v1/releases/{identifier}/package")
    def package(identifier: UUID, user: Annotated[str, Depends(principal)]) -> Response:
        return download(identifier, True)
