import json

import httpx
import pytest
from release_fixtures import admission as admission
from release_fixtures import arguments as arguments
from release_fixtures import benchmark as benchmark
from release_fixtures import records as records
from release_fixtures import setup_executor as setup_executor
from test_release_resolution import published as published

from cannery.releases.client import ReleaseClient
from cannery.releases.resolution import ResolveRequest


def reply(server, *args, **kwargs):
    result = server.request(*args, **kwargs)
    return httpx.Response(result.status_code, headers=result.headers, content=result.content)


def request(m):
    return ResolveRequest(kernel=m.kernel, runtime=m.cohort.runtime, shape=(2, 3))


def test_verified_atomic_cache_can_be_loaded_offline_and_repairs_corruption(published, tmp_path):
    _, _, _, m, _, server = published
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda req: reply(
                server,
                req.method,
                str(req.url),
                json=json.loads(req.content) if req.content else None,
            )
        )
    ) as http:
        client = ReleaseClient("http://testserver", transport=http._transport)
        saved = client.get(request(m), tmp_path)
        original = saved.package_path.read_bytes()
        assert saved.manifest_path.read_bytes() == saved.manifest.encoded()
        assert not list(tmp_path.rglob(".download-*"))
        saved.package_path.write_bytes(b"corrupt")
        assert client.get(request(m), tmp_path).package_path.read_bytes() == original

        # A valid cache avoids another package transfer, still checking live publication.
        def cached(req):
            assert not req.url.path.endswith("/package")
            return reply(
                server,
                req.method,
                str(req.url),
                json=json.loads(req.content) if req.content else None,
            )

        with httpx.Client(transport=httpx.MockTransport(cached)) as second:
            assert (
                ReleaseClient("http://testserver", transport=second._transport).get(
                    request(m), tmp_path
                )
                == saved
            )


def test_transport_retries_are_bounded_and_failed_download_leaves_no_package(published, tmp_path):
    _, _, _, m, _, server = published
    calls = []

    def handler(req):
        calls.append(req.url.path)
        if req.url.path.endswith("/package"):
            raise httpx.ReadError("network disconnected")
        return reply(
            server, req.method, str(req.url), json=json.loads(req.content) if req.content else None
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(httpx.ReadError):
            ReleaseClient("http://testserver", transport=http._transport).get(request(m), tmp_path)
    assert sum(p.endswith("/package") for p in calls) == 3
    assert not list(tmp_path.rglob("package.tar"))
    assert not list(tmp_path.rglob(".download-*"))


@pytest.mark.parametrize("mode", ["redirect", "corrupt", "oversize", "wrong-runtime"])
def test_untrusted_server_cannot_redirect_or_fill_cache_with_invalid_bytes(
    published, tmp_path, mode
):
    _, _, _, m, _, server = published

    def handler(req):
        response = reply(
            server, req.method, str(req.url), json=json.loads(req.content) if req.content else None
        )
        if req.url.path.endswith("/manifest"):
            if mode == "redirect":
                return httpx.Response(302, headers={"location": "http://other/private"})
            if mode == "corrupt":
                return httpx.Response(200, content=b"bad")
            if mode == "oversize":
                return httpx.Response(200, content=b"x" * (128 * 1024 + 1))
        return response

    selected = request(m)
    if mode == "wrong-runtime":
        selected = selected.model_copy(update={"runtime": m.cohort.runtime | {"cann": "other"}})

        def handler(req):
            # A dishonest resolver returns another runtime's release.
            content = json.loads(req.content) if req.content else None
            if content:
                content["runtime"] = m.cohort.runtime
            return reply(server, req.method, str(req.url), json=content)

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises((ValueError, httpx.HTTPStatusError)):
            ReleaseClient("http://testserver", transport=http._transport).get(selected, tmp_path)
    assert not list(tmp_path.rglob("package.tar"))


def test_url_rejects_credentials_and_queries():
    with httpx.Client():
        for url in ["file:///etc/passwd", "http://user:secret@host", "http://host?token=secret"]:
            with pytest.raises(ValueError):
                ReleaseClient(url)


def test_total_deadline_interrupts_slow_small_chunks(monkeypatch):
    import asyncio

    from cannery.releases import client as module

    original_timeout = asyncio.timeout
    monkeypatch.setattr(module.asyncio, "timeout", lambda _: original_timeout(0.02))

    class Slow(httpx.AsyncByteStream):
        async def __aiter__(self):
            for _ in range(100):
                await asyncio.sleep(0.005)
                yield b"x"

    calls = []

    def response(request):
        calls.append(request)
        return httpx.Response(200, stream=Slow())

    client = ReleaseClient("http://testserver", transport=httpx.MockTransport(response))
    with pytest.raises(TimeoutError):
        client.bytes("/api/v1/releases/kernels", 1000)
    assert len(calls) == 3


def test_compressed_response_is_rejected_before_decoding():
    import gzip

    compressed = gzip.compress(b"x" * (8 * 1024 * 1024))

    class Encoded(httpx.AsyncByteStream):
        async def __aiter__(self):
            raise AssertionError("Encoded body must never be consumed")
            yield compressed

    def response(request):
        assert request.headers["accept-encoding"] == "identity"
        return httpx.Response(200, headers={"content-encoding": "gzip"}, stream=Encoded())

    client = ReleaseClient("http://testserver", transport=httpx.MockTransport(response))
    with pytest.raises(ValueError, match="Encoded release"):
        client.bytes("/api/v1/releases/kernels", 128 * 1024)
