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


def request(m):
    return ResolveRequest(kernel=m.kernel, runtime=m.cohort.runtime, shape=(2, 3))


def test_verified_atomic_cache_can_be_loaded_offline_and_repairs_corruption(published, tmp_path):
    _, _, _, m, _, server = published
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda req: server.request(
                req.method, str(req.url), json=json.loads(req.content) if req.content else None
            )
        )
    ) as http:
        client = ReleaseClient("http://testserver", http)
        saved = client.get(request(m), tmp_path)
        original = saved.package_path.read_bytes()
        assert saved.manifest_path.read_bytes() == saved.manifest.encoded()
        assert not list(tmp_path.rglob(".download-*"))
        saved.package_path.write_bytes(b"corrupt")
        assert client.get(request(m), tmp_path).package_path.read_bytes() == original

        # A valid cache avoids another package transfer, still checking live publication.
        def cached(req):
            assert not req.url.path.endswith("/package")
            return server.request(
                req.method, str(req.url), json=json.loads(req.content) if req.content else None
            )

        with httpx.Client(transport=httpx.MockTransport(cached)) as second:
            assert ReleaseClient("http://testserver", second).get(request(m), tmp_path) == saved


def test_transport_retries_are_bounded_and_failed_download_leaves_no_package(published, tmp_path):
    _, _, _, m, _, server = published
    calls = []

    def handler(req):
        calls.append(req.url.path)
        if req.url.path.endswith("/package"):
            raise httpx.ReadError("network disconnected")
        return server.request(
            req.method, str(req.url), json=json.loads(req.content) if req.content else None
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(httpx.ReadError):
            ReleaseClient("http://testserver", http).get(request(m), tmp_path)
    assert sum(p.endswith("/package") for p in calls) == 3
    assert not list(tmp_path.rglob("package.tar"))
    assert not list(tmp_path.rglob(".download-*"))


@pytest.mark.parametrize("mode", ["redirect", "corrupt", "oversize", "wrong-runtime"])
def test_untrusted_server_cannot_redirect_or_fill_cache_with_invalid_bytes(
    published, tmp_path, mode
):
    _, _, _, m, _, server = published

    def handler(req):
        response = server.request(
            req.method, str(req.url), json=json.loads(req.content) if req.content else None
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
            return server.request(req.method, str(req.url), json=content)

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises((ValueError, httpx.HTTPStatusError)):
            ReleaseClient("http://testserver", http).get(selected, tmp_path)
    assert not list(tmp_path.rglob("package.tar"))


def test_url_rejects_credentials_and_queries():
    with httpx.Client() as http:
        for url in ["file:///etc/passwd", "http://user:secret@host", "http://host?token=secret"]:
            with pytest.raises(ValueError):
                ReleaseClient(url, http)
