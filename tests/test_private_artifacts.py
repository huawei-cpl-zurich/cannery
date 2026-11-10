from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from cannery.artifacts import ArtifactError, digest
from cannery.artifacts.factory import make_store
from cannery.artifacts.private import PrivateArtifactStore
from cannery.artifacts.service import create_app
from cannery.settings import Settings

READ = "r" * 32
WRITE = "w" * 32


def config(root):
    return Settings(
        artifact_root=root,
        artifact_read_token=SecretStr(READ),
        artifact_write_token=SecretStr(WRITE),
    )


def headers(token):
    return {"Authorization": f"Bearer {token}"}


def test_auth_immutable_concurrent_restart_and_corruption(tmp_path):
    settings = config(tmp_path)
    data = b"binary" * 1000
    checksum = digest(data)
    url = f"/v1/blobs/{checksum}"
    with TestClient(create_app(settings)) as client:
        for token in (None, READ, "wrong"):
            assert client.put(url, content=data, headers=headers(token)).status_code == 401
        with ThreadPoolExecutor(max_workers=4) as pool:
            responses = list(
                pool.map(
                    lambda _: client.put(url, content=data, headers=headers(WRITE)),
                    range(8),
                )
            )
        assert all(response.status_code == 204 for response in responses)
        assert client.get(url, headers=headers(WRITE)).status_code == 401
        assert client.get(url, headers=headers(READ)).content == data
        assert client.put(url, content=b"changed", headers=headers(WRITE)).status_code == 422
    with TestClient(create_app(settings)) as restarted:
        assert restarted.get(url, headers=headers(READ)).content == data
        (tmp_path / checksum).write_bytes(b"corrupt")
        assert restarted.get(url, headers=headers(READ)).status_code == 409
        assert restarted.put(url, content=data, headers=headers(WRITE)).status_code == 409
    assert list(tmp_path.iterdir()) == [tmp_path / checksum]


def test_partial_size_limit_and_failed_disk_leave_no_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr("cannery.artifacts.service.MAX_BYTES", 4)
    with TestClient(create_app(config(tmp_path))) as client:
        assert (
            client.put(
                f"/v1/blobs/{digest(b'large')}", content=b"large", headers=headers(WRITE)
            ).status_code
            == 413
        )
        assert (
            client.put(
                f"/v1/blobs/{digest(b'one')}", content=b"on", headers=headers(WRITE)
            ).status_code
            == 422
        )

        def fail(_):
            raise OSError("disk full")

        monkeypatch.setattr("cannery.artifacts.service.os.fsync", fail)
        assert (
            client.put(
                f"/v1/blobs/{digest(b'one')}", content=b"one", headers=headers(WRITE)
            ).status_code
            == 503
        )
    assert not list(tmp_path.iterdir())


def test_service_requires_distinct_strong_secrets(tmp_path):
    with pytest.raises(RuntimeError, match="credentials"):
        create_app(Settings(artifact_root=tmp_path))
    settings = config(tmp_path)
    settings.artifact_write_token = settings.artifact_read_token
    with pytest.raises(RuntimeError, match="distinct"):
        create_app(settings)


def test_client_integrity_disconnect_and_no_redirect_or_environment_proxy(monkeypatch):
    data = b"compiled"
    calls = []

    def handler(request):
        calls.append(request)
        assert (
            request.headers["Authorization"]
            == f"Bearer {WRITE if request.method == 'PUT' else READ}"
        )
        return httpx.Response(
            204 if request.method == "PUT" else 200,
            content=b"" if request.method == "PUT" else data,
        )

    store = PrivateArtifactStore(
        "http://private", READ, write_token=WRITE, transport=httpx.MockTransport(handler)
    )
    ref = store.put("op.so", data)
    assert store.get(ref) == data
    assert len(calls) == 3

    def broken(request):
        raise httpx.ReadTimeout("secret must not escape")

    store.transport = httpx.MockTransport(broken)
    with pytest.raises(ArtifactError, match="unavailable"):
        store.get(ref)
    for status, content in ((200, b"corrupt"), (307, b"")):
        store.transport = httpx.MockTransport(
            lambda _, status=status, content=content: httpx.Response(status, content=content)
        )
        with pytest.raises(ArtifactError):
            store.get(ref)
    store.transport = httpx.MockTransport(handler)
    with pytest.raises(ArtifactError, match="another"):
        store.get(ref.model_copy(update={"repository": "fake"}))
    monkeypatch.setattr("cannery.artifacts.private.MAX_BYTES", 2)
    with pytest.raises(ArtifactError, match="size limit"):
        store.get(ref)


def test_factory_supports_read_only_private_client(tmp_path):
    settings = Settings(artifact_backend="private", artifact_root=tmp_path)
    with pytest.raises(ArtifactError, match="credentials"):
        make_store(settings)
    settings.artifact_read_token = SecretStr(READ)
    store = make_store(settings)
    assert isinstance(store, PrivateArtifactStore)
    with pytest.raises(ArtifactError, match="writer"):
        store.put("op", b"data")


def test_retry_oversized_stored_object_is_rejected_without_reading(tmp_path, monkeypatch):
    monkeypatch.setattr("cannery.artifacts.service.MAX_BYTES", 4)
    data = b"one"
    checksum = digest(data)
    (tmp_path / checksum).write_bytes(b"oversized")
    with TestClient(create_app(config(tmp_path))) as client:
        assert (
            client.put(f"/v1/blobs/{checksum}", content=data, headers=headers(WRITE)).status_code
            == 409
        )


def test_total_deadline_cancels_trickle_and_closes_stream(monkeypatch):
    import asyncio
    import time

    closed = []

    class Trickle(httpx.AsyncByteStream):
        async def __aiter__(self):
            while True:
                await asyncio.sleep(0.005)
                yield b"a"

        async def aclose(self):
            closed.append(True)

    monkeypatch.setattr("cannery.artifacts.private.DEADLINE", 0.04)
    store = PrivateArtifactStore(
        "http://private",
        READ,
        transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=Trickle())),
    )
    from cannery.artifacts import ArtifactRef

    ref = ArtifactRef(repository="private", revision="a" * 64, sha256="a" * 64, path="op")
    start = time.monotonic()
    with pytest.raises(ArtifactError, match="unavailable"):
        store.get(ref)
    assert time.monotonic() - start < 1
    assert closed == [True]
