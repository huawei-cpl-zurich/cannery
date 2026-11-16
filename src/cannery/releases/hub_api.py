"""Small authenticated Hub read protocol over published Cannery release bytes."""

import re
from typing import Annotated, Any
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import Response
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from cannery.accounts.context import identity
from cannery.releases import catalog
from cannery.releases.download_api import release_bytes
from cannery.releases.hub import descriptor
from cannery.releases.hub import revision as commit_of
from cannery.releases.models import HubSnapshot
from cannery.settings import Settings


def missing(code: str) -> HTTPException:
    return HTTPException(404, code, headers={"X-Error-Code": code, "X-Error-Message": code})


def snapshot(db: Session, repo_id: str, revision: str) -> HubSnapshot:
    if revision.startswith("release-"):
        try:
            identifier = UUID(revision.removeprefix("release-"))
        except ValueError:
            raise missing("RevisionNotFound") from None
        row = db.get(HubSnapshot, identifier)
    elif re.fullmatch("[a-f0-9]{40}", revision):
        row = db.scalar(select(HubSnapshot).where(HubSnapshot.revision == revision))
    else:
        row = None
    if row is None or row.descriptor["repo_id"] != repo_id:
        raise missing("RevisionNotFound")
    try:
        release, _ = catalog.release_info(db, row.release_id)
    except catalog.SelectionError:
        raise missing("RevisionNotFound") from None
    if row.descriptor != descriptor(release) or row.revision != commit_of(row.descriptor):
        raise HTTPException(503, "Hub snapshot integrity failure")
    return row


def install_hub(app: FastAPI, settings: Settings, engine: Engine) -> None:
    if not settings.release_enabled or not settings.hub_enabled:
        return

    def authorized() -> str:
        return identity(settings)

    @app.get("/api/kernels/{namespace}/{name}/revision/{revision}")
    def info(
        namespace: str,
        name: str,
        revision: str,
        response: Response,
        user: Annotated[str, Depends(authorized)],
    ) -> dict[str, Any]:
        response.headers["Cache-Control"] = "private, no-store"
        with Session(engine) as db:
            row = snapshot(db, f"{namespace}/{name}", revision)
            return {
                "id": f"{namespace}/{name}",
                "sha": row.revision,
                "private": True,
                "cannery": row.descriptor,
                "siblings": [{"rfilename": "release.json"}, {"rfilename": "kernel.tar"}],
            }

    @app.api_route(
        "/kernels/{namespace}/{name}/resolve/{revision}/{filename:path}", methods=["GET", "HEAD"]
    )
    def file(
        namespace: str,
        name: str,
        revision: str,
        filename: str,
        request: Request,
        user: Annotated[str, Depends(authorized)],
    ) -> Response:
        with Session(engine) as db:
            row = snapshot(db, f"{namespace}/{name}", revision)
            identifier, commit = row.release_id, row.revision
        if filename not in {"release.json", "kernel.tar"}:
            raise missing("EntryNotFound")
        raw, _, pin = release_bytes(app, engine, identifier, filename == "kernel.tar")
        headers = {
            "X-Repo-Commit": commit,
            "ETag": '"' + pin.sha256 + '"',
            "Content-Length": str(len(raw)),
            "Cache-Control": "private, no-store",
            "Accept-Ranges": "bytes",
        }
        status = 200
        if request.method == "GET" and "range" in request.headers:
            match = re.fullmatch(r"bytes=(\d{1,20})-(\d{0,20})", request.headers["range"])
            if match is None:
                raise HTTPException(416, "Unsupported range")
            start, end = int(match[1]), int(match[2]) if match[2] else len(raw) - 1
            if start > end or end >= len(raw):
                raise HTTPException(
                    416, "Invalid range", headers={"Content-Range": f"bytes */{len(raw)}"}
                )
            headers["Content-Range"] = f"bytes {start}-{end}/{len(raw)}"
            raw = raw[start : end + 1]
            headers["Content-Length"] = str(len(raw))
            status = 206
        return Response(
            b"" if request.method == "HEAD" else raw,
            status_code=status,
            headers=headers,
            media_type="application/octet-stream"
            if filename == "kernel.tar"
            else "application/json",
        )
