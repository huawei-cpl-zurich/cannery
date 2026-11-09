"""Exercise real SDK backoff/cache locks behind the process deadline."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from filelock import FileLock

from cannery.artifacts import ArtifactError, ArtifactRef, digest
from cannery.artifacts.hf import HfArtifactStore


@pytest.mark.parametrize("mode", ["retry-after", "cache-lock", "success"])
def test_real_sdk_deadline(tmp_path, monkeypatch, mode) -> None:
    requests = []
    checksum = digest(b"binary")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_HEAD(self):
            requests.append("HEAD")
            self.send_response(200)
            self.send_header("X-Repo-Commit", "a" * 40)
            self.send_header("ETag", f'"{checksum}"')
            self.send_header("Content-Length", "6")
            self.end_headers()

        def do_GET(self):
            if self.path.startswith("/api/models/"):
                data = json.dumps(
                    {"id": "org/private", "sha": "a" * 40, "private": True, "siblings": []}
                ).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                requests.append("GET-artifact")
                if mode == "success":
                    self.send_response(200)
                    self.send_header("Content-Length", "6")
                    self.end_headers()
                    self.wfile.write(b"binary")
                    return
                self.send_response(429)
                self.send_header("Retry-After", "86400")
                self.send_header("Content-Length", "0")
                self.end_headers()

    cache = tmp_path / "cache"
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    lock_path = cache / ".locks" / "models--org--private" / f"{checksum}.lock"
    lock_path.parent.mkdir(parents=True)
    lock = FileLock(lock_path)
    if mode == "cache-lock":
        lock.acquire(timeout=1)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        store = HfArtifactStore(
            "org/private",
            "fake-token",
            timeout=3,
            endpoint=f"http://127.0.0.1:{server.server_port}",
        )
        reference = ArtifactRef(
            repository="org/private", revision="a" * 40, path="op", sha256=checksum
        )
        start = time.monotonic()
        if mode == "success":
            assert store.get(reference) == b"binary"
        else:
            with pytest.raises(ArtifactError, match="deadline exceeded"):
                store.get(reference)
        assert time.monotonic() - start < 6
        assert "HEAD" in requests
        assert ("GET-artifact" in requests) == (mode != "cache-lock")
    finally:
        lock.release()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
