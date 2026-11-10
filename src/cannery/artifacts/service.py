"""Private binary blob service. No source ingestion or code execution."""

import asyncio
import hashlib
import hmac
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse

from cannery.artifacts.private import MAX_BYTES
from cannery.settings import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    reader = settings.artifact_read_token
    writer = settings.artifact_write_token
    if not reader or not writer:
        raise RuntimeError("Private artifact service requires read and write credentials")
    read_token, write_token = reader.get_secret_value(), writer.get_secret_value()
    if len(read_token) < 32 or len(write_token) < 32 or read_token == write_token:
        raise RuntimeError("Artifact credentials must be distinct and at least 32 characters")
    root = settings.artifact_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    def authorize(value: str | None, *, write: bool = False) -> None:
        expected = f"Bearer {write_token if write else read_token}"
        if value is None or not hmac.compare_digest(value.encode(), expected.encode()):
            raise HTTPException(401, "Artifact authorization required")

    def target(checksum: str) -> Path:
        if len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
            raise HTTPException(422, "Invalid artifact digest")
        return root / checksum

    @app.put("/v1/blobs/{checksum}", status_code=204)
    async def put(
        checksum: str,
        request: Request,
        authorization: Annotated[str | None, Header()] = None,
    ) -> None:
        authorize(authorization, write=True)
        path = target(checksum)
        temporary: Path | None = None
        try:
            with NamedTemporaryFile(dir=root, delete=False) as stream:
                temporary = Path(stream.name)
                size = 0
                digest = hashlib.sha256()
                async with asyncio.timeout(30):
                    async for chunk in request.stream():
                        size += len(chunk)
                        if size > MAX_BYTES:
                            raise HTTPException(413, "Artifact exceeds size limit")
                        digest.update(chunk)
                        stream.write(chunk)
                if digest.hexdigest() != checksum:
                    raise HTTPException(422, "Artifact checksum mismatch")
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                if (
                    path.stat().st_size > MAX_BYTES
                    or hashlib.sha256(path.read_bytes()).hexdigest() != checksum
                ):
                    raise HTTPException(409, "Stored artifact integrity failure") from None
            directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except TimeoutError:
            raise HTTPException(408, "Artifact upload deadline exceeded") from None
        except OSError:
            raise HTTPException(503, "Artifact storage unavailable") from None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @app.get("/v1/blobs/{checksum}")
    def get(
        checksum: str,
        authorization: Annotated[str | None, Header()] = None,
    ) -> FileResponse:
        authorize(authorization)
        path = target(checksum)
        try:
            if not path.is_file():
                raise HTTPException(404, "Artifact unavailable")
            if path.stat().st_size > MAX_BYTES:
                raise HTTPException(409, "Stored artifact integrity failure")
            if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
                raise HTTPException(409, "Stored artifact integrity failure")
        except OSError:
            raise HTTPException(503, "Artifact storage unavailable") from None
        return FileResponse(path, media_type="application/octet-stream")

    return app
