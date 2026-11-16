import threading
import time
from contextlib import contextmanager

import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient
from huggingface_hub import HfApi, hf_hub_download
from release_fixtures import admission as admission
from release_fixtures import arguments as arguments
from release_fixtures import benchmark as benchmark
from release_fixtures import records as records
from release_fixtures import setup_executor as setup_executor
from sqlalchemy import select
from sqlalchemy.orm import Session
from test_release_resolution import published as published

from cannery.artifacts import ArtifactError, digest
from cannery.releases import state
from cannery.releases.hub_api import install_hub
from cannery.releases.models import HubSnapshot
from cannery.settings import Settings


@pytest.fixture
def hub(published):
    engine, key, store, manifest, identifier, _ = published
    app = FastAPI()
    app.state.artifacts = store
    install_hub(
        app, Settings(release_enabled=True, hub_enabled=True, dev_identity_enabled=True), engine
    )
    return published, app, TestClient(app)


def paths(manifest, identifier):
    repo = manifest.kernel
    revision = f"release-{identifier}"
    return f"/api/kernels/{repo}/revision/{revision}", f"/kernels/{repo}/resolve/{revision}"


def test_hub_pins_headers_bytes_and_resume(hub):
    (_, _, store, manifest, identifier, _), _, client = hub
    info, base = paths(manifest, identifier)
    metadata = client.get(info).json()
    assert metadata["cannery"]["manifest_sha256"]
    for name in ["release.json", "kernel.tar"]:
        head = client.head(base + "/" + name)
        full = client.get(base + "/" + name)
        assert head.status_code == full.status_code == 200
        assert not head.content
        assert int(head.headers["content-length"]) == len(full.content)
        assert head.headers["x-repo-commit"] == metadata["sha"]
        assert head.headers["etag"] == '"' + digest(full.content) + '"'
        partial = client.get(base + "/" + name, headers={"Range": "bytes=3-"})
        assert partial.status_code == 206 and partial.content == full.content[3:]
        crossing = client.get(base + "/" + name, headers={"Range": "bytes=0-999999999"})
        assert crossing.status_code == 206 and crossing.content == full.content
        assert (
            client.get(base + "/" + name, headers={"Range": "bytes=999999999-"}).status_code == 416
        )
    assert client.get(base + "/kernel.tar").content == store.get(manifest.package)
    assert client.get(base + "/missing").headers["x-error-code"] == "EntryNotFound"
    assert client.get(info.replace("release-" + str(identifier), "main")).status_code == 404
    assert client.get(info.replace(manifest.kernel, "other/kernel")).status_code == 404
    canonical = info.rsplit("/", 1)[0] + "/" + metadata["sha"]
    assert client.get(canonical).json() == metadata


def test_feature_flag_and_authentication(published):
    engine, _, store, manifest, identifier, _ = published
    info, _ = paths(manifest, identifier)
    for enabled in [False, True]:
        app = FastAPI()
        app.state.artifacts = store
        install_hub(app, Settings(release_enabled=True, hub_enabled=enabled), engine)
        assert TestClient(app).get(info).status_code == (401 if enabled else 404)


def test_withdrawal_and_backend_faults(hub, monkeypatch):
    (engine, key, store, manifest, identifier, _), _, client = hub
    info, base = paths(manifest, identifier)
    original = store.get
    monkeypatch.setattr(store, "get", lambda _: (_ for _ in ()).throw(ArtifactError("offline")))
    assert client.get(base + "/kernel.tar").status_code == 503
    monkeypatch.setattr(store, "get", original)
    assert client.get(base + "/kernel.tar").status_code == 200

    def withdrawn(ref):
        data = original(ref)
        state.withdraw(engine, key, {"namespace": "team", "role": "user"})
        return data

    monkeypatch.setattr(store, "get", withdrawn)
    assert client.get(base + "/kernel.tar").status_code == 404
    assert client.get(info).status_code == 404


@contextmanager
def serving(app):
    import socket

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(128)
        port = listener.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
        thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 10
            while not server.started:
                assert thread.is_alive() and time.monotonic() < deadline
                time.sleep(0.01)
            yield f"http://127.0.0.1:{port}"
        finally:
            server.should_exit = True
            thread.join(10)
            assert not thread.is_alive()


def test_real_huggingface_client(hub, tmp_path):
    (engine, _, _, manifest, identifier, _), app, _ = hub
    with serving(app) as endpoint:
        api = HfApi(endpoint=endpoint, token=False)
        info = api.repo_info(manifest.kernel, repo_type="kernel", revision=f"release-{identifier}")
        with Session(engine) as db:
            assert info.sha == db.scalar(select(HubSnapshot)).revision
        for name in ["release.json", "kernel.tar"]:
            path = hf_hub_download(
                manifest.kernel,
                filename=name,
                repo_type="kernel",
                revision=info.sha,
                endpoint=endpoint,
                token=False,
                cache_dir=tmp_path,
            )
            assert (
                digest(open(path, "rb").read())
                == info.cannery["manifest_sha256" if name == "release.json" else "package_sha256"]
            )
