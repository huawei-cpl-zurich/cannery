"""Bounded HTTP consumption and atomic, verified offline package caching."""

import asyncio
import builtins
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import monotonic, sleep
from typing import Any
from urllib.parse import urlsplit
from uuid import UUID

import httpx

from cannery.releases.assets import verify_package
from cannery.releases.contract import SHA, ReleaseManifest, Tolerance, read_manifest
from cannery.releases.resolution import ResolveRequest


@dataclass(frozen=True)
class CachedRelease:
    manifest: ReleaseManifest
    manifest_path: Path
    package_path: Path
    manifest_sha256: str


class ReleaseClient:
    def __init__(self, url: str, *, transport: httpx.AsyncBaseTransport | None = None):
        parsed = urlsplit(url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or any((parsed.username, parsed.password, parsed.query, parsed.fragment))
        ):
            raise ValueError("Supply a Cannery HTTP base URL without credentials or query")
        self.url = url.rstrip("/")
        self.transport = transport

    def bytes(
        self,
        path: str,
        limit: int,
        body: dict[str, Any] | None = None,
        *,
        params: dict[str, Any] | None = None,
        deadline: float | None = None,
    ) -> bytes:
        if not path.startswith("/api/v1/") or ".." in path or "?" in path:
            raise ValueError("Invalid API path")
        for attempt in range(3):
            remaining = 60.0 if deadline is None else min(60.0, deadline - monotonic())
            if remaining <= 0:
                raise TimeoutError("Release request deadline exceeded")
            try:
                return asyncio.run(self.transfer(path, limit, body, params, remaining))
            except httpx.HTTPStatusError as error:
                if error.response.status_code < 500 or attempt == 2:
                    raise
            except (httpx.TransportError, TimeoutError):
                if attempt == 2:
                    raise
            pause = 0.1 * 2**attempt
            sleep(pause if deadline is None else min(pause, max(0, deadline - monotonic())))
        raise AssertionError("Unreachable retry state")

    async def transfer(
        self,
        path: str,
        limit: int,
        body: dict[str, Any] | None,
        params: dict[str, Any] | None,
        budget: float,
    ) -> builtins.bytes:
        # The wall deadline interrupts header waits and even a continuously dripping body.
        async with asyncio.timeout(budget):
            async with httpx.AsyncClient(
                transport=self.transport, timeout=min(15, budget), follow_redirects=False
            ) as client:
                async with client.stream(
                    "GET" if body is None else "POST",
                    self.url + path,
                    json=body,
                    params=params,
                    headers={"Accept-Encoding": "identity"},
                ) as response:
                    response.raise_for_status()
                    if response.headers.get("content-encoding", "identity").lower() != "identity":
                        raise ValueError("Encoded release responses are unsupported")
                    data = bytearray()
                    async for chunk in response.aiter_bytes():
                        if len(data) + len(chunk) > limit:
                            raise ValueError("Response exceeds byte budget")
                        data.extend(chunk)
                    return bytes(data)

    def json(
        self,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        params: dict[str, Any] | None = None,
        deadline: float | None = None,
    ) -> Any:
        return json.loads(self.bytes(path, 2 * 1024 * 1024, body, params=params, deadline=deadline))

    def get(self, request: ResolveRequest, cache: Path) -> CachedRelease:
        info = self.json("/api/v1/artifacts/resolve", request.model_dump(mode="json"))
        identifier = UUID(info["id"])
        pin = info["manifest_sha256"]
        if not isinstance(pin, str) or re.fullmatch(SHA, pin) is None:
            raise ValueError("Invalid manifest pin")
        root = f"/api/v1/releases/{identifier}"
        raw = self.bytes(root + "/manifest", 128 * 1024)
        manifest = read_manifest(raw, pin)
        if (
            manifest.kernel != request.kernel
            or manifest.device != request.device
            or not manifest.supports(request.runtime, request.shape)
            or (request.language is not None and manifest.language != request.language)
            or (request.spec_version is not None and manifest.spec_version != request.spec_version)
            or (request.cohort is not None and manifest.cohort_sha256 != request.cohort)
            or (request.release_id is not None and identifier != request.release_id)
            or (
                request.rtol is not None
                and request.atol is not None
                and Tolerance(rtol=request.rtol, atol=request.atol) not in manifest.tolerances
            )
        ):
            raise ValueError("Resolved manifest does not match request")
        directory = cache / pin
        directory.mkdir(parents=True, exist_ok=True)
        package_path = directory / "package.tar"
        package = None
        if package_path.is_file() and package_path.stat().st_size <= 64 * 1024 * 1024:
            try:
                candidate = package_path.read_bytes()
                verify_package(candidate, manifest)
                package = candidate
            except ValueError:
                pass
        if package is None:
            package = self.bytes(root + "/package", 64 * 1024 * 1024)
            verify_package(package, manifest)
            atomic_write(package_path, package)
        manifest_path = directory / "manifest.json"
        atomic_write(manifest_path, raw)
        return CachedRelease(manifest, manifest_path, package_path, pin)


def atomic_write(path: Path, raw: bytes) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".download-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
