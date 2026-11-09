"""CPU-only multi-node/fault acceptance. Owns and cleans its Compose project."""

import json
import os
import socket
import subprocess
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
PROJECT = f"cannery-m0-{os.getpid()}"
with socket.socket() as listener:
    listener.bind(("127.0.0.1", 0))
    PORT = listener.getsockname()[1]
ENV = {**os.environ, "CANNERY_PORT": str(PORT), "CANNERY_DEV_IDENTITY_ENABLED": "true"}
URL = f"http://127.0.0.1:{PORT}"


def compose(*args: str, check: bool = True, timeout: int = 180, input: str | None = None) -> str:
    result = subprocess.run(
        ["docker", "compose", "-p", PROJECT, *args],
        cwd=ROOT,
        env=ENV,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        input=input,
    )
    return result.stdout


def response(path: str) -> tuple[int, bytes]:
    try:
        with urlopen(URL + path, timeout=6) as reply:
            return reply.status, reply.read()
    except HTTPError as error:
        return error.code, error.read()


def ready() -> None:
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        try:
            if response("/health/ready")[0] == 200:
                return
        except (URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(0.5)
    raise AssertionError("Readiness recovery deadline exceeded")


def python(source: str) -> str:
    return compose("exec", "-T", "api", "python", "-c", source)


def main() -> None:
    try:
        compose("build", timeout=300)
        compose("up", "-d", "--scale", "worker=2", "--wait", timeout=180)
        compose("run", "--rm", "migrate")  # A second migration is harmless.
        ready()
        assert response("/health/live")[0] == 200
        for path in (
            "/global-best",
            "/my-submissions",
            "/register-kernel",
            "/submit-implementation",
        ):
            assert response(path)[0] == 200
        assert response("/admin")[0] == 403
        assert response("/static/htmx.min.js")[0] == 200
        compose("exec", "-T", "api", "cannery", "health")
        python("""
from cannery.artifacts.factory import make_store
from cannery.settings import Settings
store = make_store(Settings())
reference = store.put('probe.bin', b'cpu-only')
assert store.get(reference) == b'cpu-only'
""")
        python("""
from cannery.git_intake.transport import RemoteRule, pinned_environment
rule = RemoteRule(url='https://localhost/repo.git', namespaces=('team',),
                  allowed_networks=('127.0.0.0/8', '::1/128'))
assert pinned_environment(rule)['CANNERY_PINNED_ADDRESS']
""")
        compose(
            "exec",
            "-T",
            "api",
            "python",
            "-",
            input=(ROOT / "scripts/check_frozen_specs.py").read_text(),
        )
        # No consumers: durable queued work must survive a lost dispatch and a
        # Redis restart. The reconciler, rather than the caller, replays the ID.
        compose("stop", "worker", "reconciler")
        key = python("""
from cannery.database import make_engine
from cannery.jobs import create_probe
from cannery.settings import Settings
engine = make_engine(Settings())
print(create_probe(engine))
engine.dispose()
""").strip()
        compose("stop", "redis")
        started = time.monotonic()
        code, body = response("/health/ready")
        assert code == 503 and not json.loads(body)["checks"]["redis"]
        assert time.monotonic() - started < 6
        assert response("/health/live")[0] == 200
        compose("start", "redis")
        ready()
        compose("start", "worker", "reconciler")
        python(f"""
import time
from sqlalchemy import select
from cannery.database import make_engine
from cannery.jobs import ProbeJob
from cannery.settings import Settings
from cannery.worker import probe
from uuid import UUID
engine = make_engine(Settings())
key = UUID({key!r})
for _ in range(40):
    with engine.connect() as connection:
        row = connection.execute(select(ProbeJob).where(ProbeJob.id == key)).one()
    if row.status == 'complete':
        assert row.effects == 1
        break
    time.sleep(.5)
else:
    raise AssertionError('Durable worker recovery deadline exceeded')
for _ in range(4):
    probe.send(str(key))
time.sleep(1)
with engine.connect() as connection:
    row = connection.execute(select(ProbeJob).where(ProbeJob.id == key)).one()
    assert row.status == 'complete' and row.effects == 1
engine.dispose()
""")
        # A paused server accepts TCP but cannot answer. A client deadline must
        # cover the established session, while liveness remains responsive.
        compose("pause", "postgres")
        try:
            started = time.monotonic()
            code, body = response("/health/ready")
            assert code == 503 and not json.loads(body)["checks"]["postgres"]
            assert time.monotonic() - started < 6
            assert response("/health/live")[0] == 200
        finally:
            compose("unpause", "postgres")
        ready()
        print("PASS: boot/migrations, web/CLI, fake store, two workers, outages and recovery")
    except BaseException:
        print(compose("logs", "--no-color", "--tail", "60", check=False, timeout=15))
        raise
    finally:
        compose("unpause", "postgres", "redis", check=False, timeout=15)
        compose("down", "--volumes", "--remove-orphans", "--rmi", "local", timeout=60)


if __name__ == "__main__":
    main()
