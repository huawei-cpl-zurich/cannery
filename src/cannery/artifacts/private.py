"""Bounded transport for the private, content-addressed artifact service."""

import asyncio

import httpx

from cannery.artifacts import ArtifactError, ArtifactRef, digest, verified

MAX_BYTES = 64 * 1024 * 1024
DEADLINE = 30


class PrivateArtifactStore:
    def __init__(
        self,
        url: str,
        token: str,
        *,
        write_token: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.url = url.rstrip("/")
        self.token = token
        self.write_token = write_token
        self.transport = transport

    async def _transfer(self, method: str, checksum: str, data: bytes | None) -> bytes:
        token = self.write_token if method == "PUT" else self.token
        async with httpx.AsyncClient(
            transport=self.transport,
            timeout=DEADLINE,
            follow_redirects=False,
            trust_env=False,
        ) as client:
            async with client.stream(
                method,
                f"{self.url}/v1/blobs/{checksum}",
                content=data,
                headers={"Authorization": f"Bearer {token}"},
            ) as response:
                response.raise_for_status()
                result = bytearray()
                async for chunk in response.aiter_bytes():
                    if len(result) + len(chunk) > MAX_BYTES:
                        raise ArtifactError("Artifact exceeds size limit")
                    result.extend(chunk)
                return bytes(result)

    async def _bounded(self, method: str, checksum: str, data: bytes | None) -> bytes:
        async with asyncio.timeout(DEADLINE):
            return await self._transfer(method, checksum, data)

    def _request(self, method: str, checksum: str, data: bytes | None = None) -> bytes:
        # ArtifactStore is synchronous and used by trusted worker/coordinator code.
        try:
            return asyncio.run(self._bounded(method, checksum, data))
        except (httpx.HTTPError, TimeoutError):
            raise ArtifactError(
                "Private artifact service unavailable or request rejected"
            ) from None

    def put(self, path: str, data: bytes) -> ArtifactRef:
        if not self.write_token:
            raise ArtifactError("Artifact writer credentials required")
        if len(data) > MAX_BYTES:
            raise ArtifactError("Artifact exceeds size limit")
        checksum = digest(data)
        reference = ArtifactRef(
            repository="private",
            revision=checksum,
            path=path,
            sha256=checksum,
        )
        self._request("PUT", checksum, data)
        verified(self._request("GET", checksum), reference)
        return reference

    def get(self, reference: ArtifactRef) -> bytes:
        if reference.repository != "private" or reference.revision != reference.sha256:
            raise ArtifactError("Artifact belongs to another repository or revision")
        return verified(self._request("GET", reference.sha256), reference)
