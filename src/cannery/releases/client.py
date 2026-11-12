"""Bounded HTTP consumption and atomic, verified offline package caching."""

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
    def __init__(self, url: str, client: httpx.Client):
        parsed = urlsplit(url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or any((parsed.username, parsed.password, parsed.query, parsed.fragment))
        ):
            raise ValueError("Supply a Cannery HTTP base URL without credentials or query")
        self.url = url.rstrip("/")
        self.client = client

    def bytes(
        self,
        path: str,
        limit: int,
        body: dict[str, Any] | None = None,
        *,
        params: dict[str, Any] | None = None,
    ) -> bytes:
        if not path.startswith("/api/v1/") or ".." in path or "?" in path:
            raise ValueError("Invalid API path")
        for attempt in range(3):
            try:
                with self.client.stream(
                    "GET" if body is None else "POST",
                    self.url + path,
                    json=body,
                    params=params,
                    timeout=15,
                    follow_redirects=False,
                ) as response:
                    response.raise_for_status()
                    deadline = monotonic() + 60
                    data = bytearray()
                    for chunk in response.iter_bytes(65536):
                        if monotonic() > deadline or len(data) + len(chunk) > limit:
                            raise ValueError("Response exceeds byte budget")
                        data.extend(chunk)
                    return bytes(data)
            except httpx.HTTPStatusError as error:
                if error.response.status_code < 500 or attempt == 2:
                    raise
            except httpx.TransportError:
                if attempt == 2:
                    raise
            sleep(0.1 * 2**attempt)
        raise AssertionError("Unreachable retry state")

    def json(
        self,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        params: dict[str, Any] | None = None,
    ) -> Any:
        return json.loads(self.bytes(path, 2 * 1024 * 1024, body, params=params))

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
