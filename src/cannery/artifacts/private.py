"""Bounded transport for the private, content-addressed artifact service."""

import httpx

from cannery.artifacts import ArtifactError, ArtifactRef, digest, verified

MAX_BYTES = 64 * 1024 * 1024


class PrivateArtifactStore:
    def __init__(
        self, url: str, token: str, *, write_token: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.url = url.rstrip("/")
        self.token = token
        self.write_token = write_token
        self.transport = transport

    def _request(self, method: str, checksum: str, data: bytes | None = None) -> bytes:
        try:
            with httpx.Client(
                transport=self.transport, timeout=30, follow_redirects=False,
                trust_env=False,
            ) as client:
                with client.stream(
                    method, f"{self.url}/v1/blobs/{checksum}", content=data,
                    headers={"Authorization": f"Bearer {self.write_token if method == 'PUT' else self.token}"},
                ) as response:
                    response.raise_for_status()
                    result = bytearray()
                    for chunk in response.iter_bytes():
                        result.extend(chunk)
                        if len(result) > MAX_BYTES:
                            raise ArtifactError("Artifact exceeds size limit")
                    return bytes(result)
        except httpx.HTTPError:
            raise ArtifactError("Private artifact service unavailable or request rejected") from None

    def put(self, path: str, data: bytes) -> ArtifactRef:
        if not self.write_token:
            raise ArtifactError("Artifact writer credentials required")
        if len(data) > MAX_BYTES:
            raise ArtifactError("Artifact exceeds size limit")
        checksum = digest(data)
        reference = ArtifactRef(
            repository="private", revision=checksum, path=path, sha256=checksum,
        )
        self._request("PUT", checksum, data)
        # Successful admission requires independent read-back, including after retries.
        verified(self._request("GET", checksum), reference)
        return reference

    def get(self, reference: ArtifactRef) -> bytes:
        if reference.repository != "private" or reference.revision != reference.sha256:
            raise ArtifactError("Artifact belongs to another repository or revision")
        return verified(self._request("GET", reference.sha256), reference)
