"""Real PostgreSQL/Redis/HF/fork integration; native execution is CPU mocked."""

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from release_fixtures import admission as admission
from release_fixtures import arguments as arguments
from release_fixtures import benchmark as benchmark
from release_fixtures import setup_executor as setup_executor
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from test_auth_docker import docker, free_port, until
from test_hub_api import serving
from test_release_resolution import published as published

from cannery.accounts import operator
from cannery.accounts.api import install_auth
from cannery.accounts.context import install_context
from cannery.database import Base
from cannery.registry.models import Kernel, SpecVersion
from cannery.releases.hub import ensure_snapshot
from cannery.releases.hub_api import install_hub
from cannery.releases.models import HubSnapshot, Release
from cannery.settings import Settings

pytestmark = pytest.mark.skipif(
    os.environ.get("CANNERY_HUB_DOCKER") != "1", reason="Set CANNERY_HUB_DOCKER=1"
)
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def services():
    slug = "cannery-m8-" + uuid4().hex[:10]
    pg, redis, network = slug + "-pg", slug + "-redis", slug + "-net"
    try:
        docker("network", "create", "--label", "cannery.task=m8", network)
        for name, image, port, extra in [
            (
                pg,
                "postgres:17",
                5432,
                [
                    "-e",
                    "POSTGRES_PASSWORD=fixture",
                    "-e",
                    "POSTGRES_USER=cannery",
                    "-e",
                    "POSTGRES_DB=cannery",
                ],
            ),
            (redis, "redis:7-alpine", 6379, []),
        ]:
            docker(
                "run",
                "-d",
                "--name",
                name,
                "--network",
                network,
                "--label",
                "cannery.task=m8",
                "-p",
                f"127.0.0.1:{free_port()}:{port}",
                *extra,
                image,
            )
        pgport = docker("port", pg, "5432").rsplit(":", 1)[1]
        redisport = docker("port", redis, "6379").rsplit(":", 1)[1]
        until(lambda: docker("exec", pg, "pg_isready", "-h", "127.0.0.1", "-U", "cannery"))
        yield (
            f"postgresql+psycopg://cannery:fixture@127.0.0.1:{pgport}/cannery",
            f"redis://127.0.0.1:{redisport}/0",
            pg,
            redis,
        )
    finally:
        for name in [redis, pg]:
            subprocess.run(["docker", "rm", "-f", "-v", name], timeout=30, capture_output=True)
        subprocess.run(["docker", "network", "rm", network], timeout=30, capture_output=True)


@pytest.fixture
def records(services):
    engine = create_engine(services[0], connect_args={"connect_timeout": 3})
    Base.metadata.create_all(engine)
    key = uuid4()
    with Session(engine) as db, db.begin():
        db.add(Kernel(id="kernel", namespace="team", name="identity"))
        db.flush()
        db.add(
            SpecVersion(
                id=key,
                kernel_id="kernel",
                version=1,
                content_sha256="a" * 64,
                manifest={"kernel": {"namespace": "team", "name": "identity"}},
                snapshot={},
                file_sha256={},
                smoke={},
            )
        )
    yield engine, key
    engine.dispose()


def test_hub_client_postgres_redis_failures_and_revocation(published, services, tmp_path):
    engine, _, store, manifest, identifier, _ = published
    client_python = os.environ["CANNERY_HUB_CLIENT_PYTHON"]
    assert Path(client_python).is_file()
    operator.create(engine, "alice", "fixture account password")
    operator.membership(engine, "alice", "team", grant=True)
    settings = Settings(
        auth_enabled=True,
        auth_cookie_secure=False,
        auth_origin="http://127.0.0.1",
        release_enabled=True,
        hub_enabled=True,
        redis_url=services[1],
    )
    app = FastAPI()
    app.state.artifacts = store
    install_auth(app, settings, engine)
    install_context(app, settings, engine)
    install_hub(app, settings, engine)
    password = tmp_path / "password"
    password.write_text("fixture account password")
    password.chmod(0o600)
    env = os.environ | {"HOME": str(tmp_path / "home"), "PYTHONPATH": str(ROOT / "src")}
    env.pop("HF_TOKEN", None)
    env.pop("LOCAL_KERNELS", None)
    with serving(app) as endpoint:
        env.update(HF_ENDPOINT=endpoint, KERNELS_CACHE=str(tmp_path / "cache"))
        cli = [sys.executable, "-c", "from cannery.cli import app; app()"]
        login = subprocess.run(
            cli
            + ["login", "--url", endpoint, "--username", "alice", "--password-file", str(password)],
            env=env,
            capture_output=True,
            timeout=20,
        )
        assert login.returncode == 0
        env.update(HTTP_PROXY="http://127.0.0.1:1", HTTPS_PROXY="http://127.0.0.1:1", NO_PROXY="")
        code = (
            "from types import SimpleNamespace; import cannery_kernels; "
            "cannery_kernels.load_cannery_package="
            "lambda *a,**k:SimpleNamespace(forward=lambda x:x+1); "
            "from kernels import get_kernel; import sys; "
            "k=get_kernel(sys.argv[1],revision=sys.argv[2],backend='cann',trust_remote_code=True); "
            "assert k.forward(2)==3"
        )
        command = [client_python, "-c", code, manifest.kernel, "release-" + str(identifier)]

        def load(ok=True):
            result = subprocess.run(command, env=env, capture_output=True, timeout=70)
            assert (result.returncode == 0) == ok, result.stderr[-500:]

        assert (
            httpx.get(
                endpoint + f"/api/kernels/{manifest.kernel}/revision/release-{identifier}",
                trust_env=False,
            ).status_code
            == 401
        )
        load()
        load()  # Warm cache still authenticates and revalidates publication.
        docker("stop", services[3])
        load()  # Existing tokens survive Redis failure.
        denied = subprocess.run(
            cli
            + ["login", "--url", endpoint, "--username", "alice", "--password-file", str(password)],
            env=env,
            capture_output=True,
            timeout=20,
        )
        assert denied.returncode != 0
        docker("start", services[3])
        docker("stop", services[2])
        load(False)
        docker("start", services[2])
        until(lambda: docker("exec", services[2], "pg_isready", "-h", "127.0.0.1", "-U", "cannery"))
        engine.dispose()
        load()
        operator.configure(engine, "alice", enabled=False)
        load(False)  # A warm HF cache cannot bypass credential revocation.


def test_postgres_concurrent_snapshot_creation_is_idempotent(published):
    engine, _, _, _, identifier, _ = published
    with engine.begin() as connection:
        connection.execute(HubSnapshot.__table__.delete())

    def create(_):
        with Session(engine) as db, db.begin():
            return ensure_snapshot(db, db.get(Release, identifier)).revision

    with ThreadPoolExecutor(max_workers=4) as pool:
        commits = list(pool.map(create, range(4)))
    assert len(set(commits)) == 1
    with Session(engine) as db:
        assert len(list(db.scalars(select(HubSnapshot)))) == 1
