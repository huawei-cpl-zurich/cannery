"""Real private-Git registration, two registrars, durable faults and retained rebuild."""

import json
import os
import socket
import subprocess
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
PROJECT = f"cannery-m1-{uuid4().hex}"
with socket.socket() as listener:
    listener.bind(("127.0.0.1", 0))
    PORT = listener.getsockname()[1]
ENV = {
    **os.environ,
    "COMPOSE_PROJECT_NAME": PROJECT,
    "CANNERY_PORT": str(PORT),
    "CANNERY_DEV_IDENTITY_ENABLED": "true",
}
URL = f"http://127.0.0.1:{PORT}"


def compose(*args, timeout=180, check=True):
    return subprocess.run(
        [
            "docker",
            "compose",
            "-p",
            PROJECT,
            "-f",
            "compose.yaml",
            "-f",
            "compose.registration.yaml",
            *args,
        ],
        cwd=ROOT,
        env=ENV,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=check,
    ).stdout


def execute(service, code):
    return compose("exec", "-T", service, "python", "-c", code)


def request(path, body=None):
    try:
        with urlopen(
            Request(
                URL + path,
                data=None if body is None else json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            ),
            timeout=6,
        ) as reply:
            return reply.status, json.load(reply)
    except HTTPError as error:
        return error.code, json.load(error)


def wait_ready():
    for _ in range(60):
        try:
            if request("/health/ready")[0] == 200:
                return
        except (URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(0.5)
    raise AssertionError("readiness deadline exceeded")


def submit(sha, **changes):
    source = {
        "url": "https://git-fixture:8443/repo.git",
        "commit": sha,
        "path": "spec/",
        "dependencies": ["shared"],
    } | changes
    status, result = request("/api/v1/kernels", {"source": source})
    assert status == 202, result
    return result["id"]


def outcome(key, expected="complete"):
    deadline = time.monotonic() + 100
    while time.monotonic() < deadline:
        status, result = request(f"/api/v1/registrations/{key}")
        assert status == 200
        if result["status"] in ("complete", "failed"):
            assert result["status"] == expected, result
            return result
        time.sleep(0.5)
    raise AssertionError(f"registration {key} did not finish")


def main():
    try:
        compose("build", timeout=600)
        compose("up", "-d", "--scale", "registrar=2", "--wait", timeout=180)
        wait_ready()
        revisions = json.loads(
            compose("exec", "-T", "git-fixture", "cat", "/fixtures/revisions.json")
        )
        print("Services ready; exercising private Git registration", flush=True)
        # Real public CLI intake and private HTTPS smart Git with scoped credentials.
        compose(
            "exec",
            "-T",
            "api",
            "cannery",
            "register",
            "--url",
            "http://api:8000",
            "--repo",
            "https://git-fixture:8443/repo.git",
            "--commit",
            revisions["v1"],
            "--path",
            "spec/",
            "--dependency",
            "shared",
            "--wait",
            timeout=120,
        )
        first = request("/api/v1/kernels/team/identity/specs/1")[1]
        assert first["smoke"]["assurance"] == "diagnostic"
        one, two = submit(revisions["v1"]), submit(revisions["v1"])
        assert outcome(one)["spec"]["id"] == outcome(two)["spec"]["id"] == first["id"]
        print("Duplicate jobs froze one immutable version", flush=True)
        # Admin-only semantic version update; version1 remains byte-for-byte stable.
        assert outcome(submit(revisions["v2"]), "failed")["error"] == "spec_conflict"
        key = json.loads(
            execute(
                "api",
                f"""
import httpx
body={{'source':{{'url':'https://git-fixture:8443/repo.git','commit':{revisions["v2"]!r},'path':'spec','dependencies':['shared']}}}}
r=httpx.post('http://admin-api:8000/api/v1/kernels',json=body,timeout=6)
r.raise_for_status(); print(r.text)
""",
            )
        )["id"]
        assert outcome(key)["spec"]["version"] == 2
        assert request("/api/v1/kernels/team/identity/specs/1")[1] == first
        assert request("/api/v1/kernels/other/identity/specs/1")[0] == 404
        assert (
            request("/api/v1/kernels", {"namespace": "other", "source": first["source"]})[0] == 403
        )
        # Unconfigured remote fails before DNS; missing paths/commits and empty cases are bounded.
        assert (
            outcome(
                submit(revisions["v1"], url="https://must-not-resolve.invalid/repo.git"), "failed"
            )["error"]
            == "repository_denied"
        )
        outcome(submit(revisions["v1"], path="absent"), "failed")
        outcome(submit("f" * 40), "failed")
        assert outcome(submit(revisions["bad"]), "failed")["error"] == "reference_failed"
        print("Version, authorization and invalid-source checks passed", flush=True)
        # Lost dispatch survives a Redis outage and is reconciled from PostgreSQL.
        compose("stop", "redis")
        queued = submit(revisions["v1"])
        assert request(f"/api/v1/registrations/{queued}")[1]["status"] == "queued"
        compose("start", "redis")
        wait_ready()
        outcome(queued)
        # PostgreSQL outages return safe503 and recover after restart.
        compose("stop", "postgres")
        assert request("/api/v1/registrations")[0] == 503
        compose("start", "postgres")
        wait_ready()
        # An abandoned claimed attempt is recovered after worker restart/lease expiry.
        compose("stop", "registrar", "reconciler")
        stale = submit(revisions["v1"])
        execute(
            "api",
            f"""
from datetime import UTC,datetime,timedelta
from uuid import UUID
from sqlalchemy.orm import Session
from cannery.database import make_engine
from cannery.settings import Settings
from cannery.registry.registrations import Registration,claim_registration,prepare_handle
engine=make_engine(Settings())
claim=claim_registration(engine,UUID({stale!r}))
prepare_handle(engine,claim)
with Session(engine) as s,s.begin():
    s.get(Registration,claim.id).lease_until=datetime.now(UTC)-timedelta(seconds=1)
engine.dispose()
""",
        )
        compose("start", "registrar", "reconciler")
        outcome(stale)
        print("Database, dispatch and lease recovery passed", flush=True)
        # Remove external repository, GC retained Git, and rerun CPU smoke from protected snapshot.
        execute(
            "git-fixture",
            "import shutil; shutil.rmtree('/fixtures/repo.git'); shutil.rmtree('/fixtures/work')",
        )
        execute(
            "registrar",
            """
from sqlalchemy import select
from sqlalchemy.orm import Session
from cannery.database import make_engine
from cannery.settings import Settings
from cannery.registry.models import SpecVersion
from cannery.git_intake import SnapshotBinding
from cannery.git_intake.process import git
from cannery.git_intake.snapshots import SnapshotStore
from cannery.registry.manifest import validate_spec
from cannery.registry.sandbox import run_smoke
config=Settings(); engine=make_engine(config); store=SnapshotStore(config.snapshot_root)
with Session(engine) as session:
    versions=list(session.scalars(select(SpecVersion)))
    assert len(versions)==2
    for version in versions:
        binding=SnapshotBinding.model_validate(version.snapshot)
        repository=store.root / (binding.internal_repository+'.git')
        git(repository,'gc','--prune=now')
        retained=store.read(binding).files
        assert 'unrelated.txt' not in retained
        spec=validate_spec(retained,binding.source.path)
        assert run_smoke(spec,retained,binding.source.path,image=config.smoke_image).cases > 0
engine.dispose()
""",
        )
        print(
            "PASS: private HTTPS Git, CLI/API, two registrars, versions, outages, "
            "lease recovery, retained CPU rebuild"
        )
    except BaseException:
        print(compose("logs", "--tail", "30", check=False), flush=True)
        raise
    finally:
        compose("down", "--volumes", "--remove-orphans", "--rmi", "local", timeout=180)


if __name__ == "__main__":
    main()
