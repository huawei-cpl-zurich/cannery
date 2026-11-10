"""M2 user journey and deterministic service/worker failures; no candidate execution."""

import json
import re
import time
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen
from uuid import uuid4

import registration_acceptance as harness

harness.PROJECT = f"cannery-m2-{uuid4().hex}"
harness.ENV["COMPOSE_PROJECT_NAME"] = harness.PROJECT
REPO = "https://git-fixture:8443/repo.git"


def revision(number: int, *, entrypoint: str = "forward", adapter: str = "catlass") -> str:
    manifest = {
        "schema_version": 1,
        "kernel": "team/identity",
        "spec_version": 1,
        "implementation": {
            "device": "a5",
            "language": "catlass",
            "entrypoint": f"kernel.py:{entrypoint}",
            "build": {"adapter": adapter, "inputs": ["kernel.py"], "artifact": "identity_a5"},
        },
        "validation": {"requested_rtol": 0.001, "requested_atol": 0.0001},
    }
    return harness.execute(
        "git-fixture",
        f"""
from pathlib import Path
import subprocess
root=Path('/fixtures/work'); (root/'candidate').mkdir(exist_ok=True)
(root/'candidate/submission.yaml').write_text({json.dumps(manifest)!r})
(root/'candidate/kernel.py').write_text(
    "raise RuntimeError('M2 must not execute author code')\\n"
    "# revision {number}\\n")
def command(*args):
    return subprocess.check_output(args,stderr=subprocess.DEVNULL,timeout=20).decode().strip()
command('git','-C',str(root),'add','candidate')
command('git','-C',str(root),'-c','user.name=fixture','-c','user.email=x@y','commit','-m','candidate{number}')
command('git','-C',str(root),'push','/fixtures/repo.git','HEAD:refs/heads/candidates')
print(command('git','-C',str(root),'rev-parse','HEAD'))
""",
    ).strip()


def submit(commit: str, parent: str | None = None, **changes: str) -> str:
    if parent:
        status, result = harness.request(
            f"/api/v1/submissions/{parent}/revisions", {"commit": commit}
        )
    else:
        status, result = harness.request(
            "/api/v1/kernels/team/identity/submissions",
            {
                "spec_version": 1,
                "device": "a5",
                "language": "catlass",
                "source": {"url": REPO, "commit": commit, "path": "candidate"} | changes,
            },
        )
    assert status == 202, result
    return result["id"]


def retained(identifier: str, expected: str = "awaiting_build") -> dict:
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        code, body = harness.request(f"/api/v1/submissions/{identifier}")
        assert code == 200, body
        if body["status"] in ("awaiting_build", "failed"):
            assert body["status"] == expected, body
            assert body["eta_seconds"] is None
            return body
        time.sleep(0.5)
    raise AssertionError(f"Submission {identifier} did not finish source intake")


def cli(*args: str) -> str:
    output = harness.compose("exec", "-T", "api", "cannery", *args, timeout=180)
    found = re.search(r"Submission ([a-f0-9-]{36})", output)
    assert found and "awaiting build" in output and "ETA unknown" in output, output
    return found[1]


def web_request(path: str, fields: dict[str, str]) -> str:
    request = Request(
        harness.URL + path,
        data=urlencode(fields).encode(),
        headers={"Origin": harness.URL, "Content-Type": "application/x-www-form-urlencoded"},
    )
    with urlopen(request, timeout=8) as response:
        assert response.status == 200
        identifier = urlsplit(response.geturl()).path.rsplit("/", 1)[-1]
        assert re.fullmatch(r"[a-f0-9-]{36}", identifier), response.geturl()
    return identifier


def abandon(identifier: str) -> str:
    token = harness.execute(
        "api",
        f"""
from datetime import UTC,datetime,timedelta
from uuid import UUID
from sqlalchemy.orm import Session
from cannery.database import make_engine
from cannery.settings import Settings
from cannery.submissions.state import IntakeState,claim_intake,prepare_scratch
engine=make_engine(Settings()); claim=claim_intake(engine,UUID({identifier!r}))
token=prepare_scratch(engine,claim)
with Session(engine) as session,session.begin():
    session.get(IntakeState,claim.id).lease_until=datetime.now(UTC)-timedelta(seconds=1)
print(token); engine.dispose()
""",
    ).strip()
    harness.compose(
        "run",
        "--rm",
        "--no-deps",
        "registrar",
        "python",
        "-c",
        f"""
from pathlib import Path
from uuid import UUID
root=Path('/var/lib/cannery/snapshots/.submission-scratch')
path=root / (UUID({identifier!r}).hex+'-'+UUID({token!r}).hex)
path.mkdir(parents=True); (path/'abandoned').write_text('recover me')
""",
    )
    return token


def verify_retained(first: str, second: str) -> None:
    harness.execute(
        "registrar",
        f"""
from uuid import UUID
from sqlalchemy import delete,select,update
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session
from cannery.database import make_engine
from cannery.settings import Settings
from cannery.git_intake import SnapshotBinding
from cannery.git_intake.process import git
from cannery.git_intake.snapshots import SnapshotStore
from cannery.submissions.manifest import validate_submission
from cannery.submissions.models import Candidate,Submission
from cannery.submissions.state import IntakeState
config=Settings(); engine=make_engine(config); store=SnapshotStore(config.snapshot_root)
with Session(engine) as session:
    old,new=session.get(Candidate,UUID({first!r})),session.get(Candidate,UUID({second!r}))
    assert old.binding==new.binding and old.binding_id==new.binding_id==old.id
    assert old.content_sha256!=new.content_sha256
    assert session.get(Submission,new.id).parent_id==old.id
    for candidate in session.scalars(select(Candidate)):
        binding=SnapshotBinding.model_validate(candidate.snapshot)
        git(store.root/(binding.internal_repository+'.git'),'gc','--prune=now')
        files=store.read(binding).files
        assert set(files)=={{'candidate/kernel.py','candidate/submission.yaml'}}
        validated=validate_submission(files,binding.source.path)
        assert validated.content_sha256==candidate.content_sha256
    pending=select(IntakeState).where(IntakeState.scratch_token.is_not(None))
    assert session.scalars(pending).first() is None
assert not list((config.snapshot_root/'.submission-scratch').iterdir())
for statement in (update(Candidate).where(Candidate.id==UUID({first!r})).values(binding={{}}),
                  delete(Candidate).where(Candidate.id==UUID({first!r}))):
    try:
        with engine.begin() as connection: connection.execute(statement)
    except DBAPIError as error: assert getattr(error.orig,'sqlstate',None)=='P0001'
    else: raise AssertionError('Accepted candidate mutation')
engine.dispose()
""",
    )


def main() -> None:
    try:
        harness.compose("build", timeout=600)
        harness.compose("up", "-d", "--scale", "registrar=2", "--wait")
        harness.wait_ready()
        revisions = json.loads(
            harness.compose("exec", "-T", "git-fixture", "cat", "/fixtures/revisions.json")
        )
        harness.outcome(harness.submit(revisions["v1"]))
        sha1 = revision(1)
        first = cli(
            "submit",
            "team/identity",
            "--url",
            "http://api:8000",
            "--repo",
            REPO,
            "--commit",
            sha1,
            "--path",
            "candidate",
            "--spec-version",
            "1",
            "--device",
            "a5",
            "--lang",
            "catlass",
            "--wait",
        )
        before = retained(first)
        sha2 = revision(2)
        second = cli("revise", first, "--url", "http://api:8000", "--commit", sha2, "--wait")
        child = retained(second)
        assert child["binding_id"] == before["binding_id"] == first and child["parent_id"] == first
        assert child["candidate"]["content_sha256"] != before["candidate"]["content_sha256"]
        assert harness.request(f"/api/v1/submissions/{first}")[1] == before
        assert child["stages"]["build"] == "pending"
        # Repeated deliveries must not create replacement candidates or change parent provenance.
        harness.execute(
            "api",
            f"""
from cannery.registry.dispatch import publish_submission
from cannery.settings import Settings
for _ in range(8): publish_submission(Settings().redis_url.get_secret_value(),{first!r})
""",
        )
        assert retained(first) == before
        print("PASS: CLI submit/revise, immutable binding, honest pending stages", flush=True)
        spec = harness.request("/api/v1/kernels/team/identity/specs/1")[1]
        web_first = web_request(
            "/submit-implementation",
            {
                "spec_id": spec["id"],
                "device": "a5",
                "language": "catlass",
                "url": REPO,
                "commit": sha1,
                "path": "candidate",
                "dependencies": "",
            },
        )
        retained(web_first)
        web_next = web_request(f"/submissions/{web_first}/revisions", {"commit": sha2})
        assert retained(web_next)["parent_id"] == web_first
        with urlopen(harness.URL + "/my-submissions", timeout=6) as response:
            html = response.read().decode()
            assert first in html and web_first in html and "awaiting_build" in html
        print("PASS: real same-origin web submission, revision and owned status pages", flush=True)
        # Typed manifest and frozen-binding errors never publish a candidate.
        assert (
            retained(submit(revision(3, entrypoint="other"), first), "failed")["error"]
            == "binding_mismatch"
        )
        assert (
            retained(submit(revision(4, adapter="shell")), "failed")["error"]
            == "invalid_submission"
        )
        denied = submit(sha1, url="https://must-not-resolve.invalid/repo.git")
        assert retained(denied, "failed")["error"] == "repository_denied"
        retained(submit(sha1, path="missing"), "failed")
        retained(submit("f" * 40), "failed")
        assert (
            harness.request(
                f"/api/v1/submissions/{first}/revisions", {"commit": sha2, "device": "a3"}
            )[0]
            == 422
        )
        # A disconnected Git node causes a retry; the same request recovers after restart.
        harness.compose("stop", "git-fixture")
        recovering = submit(sha1)
        for _ in range(40):
            if harness.request(f"/api/v1/submissions/{recovering}")[1]["error"]:
                break
            time.sleep(0.5)
        else:
            raise AssertionError("Git outage did not exercise a failed attempt")
        harness.compose("start", "git-fixture")
        retained(recovering)
        harness.compose("stop", "redis")
        queued = submit(sha1)
        assert harness.request(f"/api/v1/submissions/{queued}")[1]["status"] == "queued"
        harness.compose("start", "redis")
        harness.wait_ready()
        retained(queued)
        harness.compose("stop", "postgres")
        assert harness.request("/api/v1/submissions/mine")[0] == 503
        harness.compose("start", "postgres")
        harness.wait_ready()
        print("PASS: invalid sources, Git/Redis/PostgreSQL outage recovery", flush=True)
        harness.compose("stop", "registrar", "reconciler")
        stale = submit(sha1)
        abandon(stale)
        harness.compose("start", "registrar", "reconciler")
        retained(stale)
        assert harness.request(f"/api/v1/submissions/{first}")[1] == before
        # Re-read protected source after deleting external Git and its work checkout.
        harness.execute(
            "git-fixture",
            "import shutil; shutil.rmtree('/fixtures/repo.git'); shutil.rmtree('/fixtures/work')",
        )
        verify_retained(first, second)
        print(
            "PASS: stale-lease/scratch recovery, retained snapshots after upstream deletion and GC"
        )
    except BaseException:
        print(harness.compose("logs", "--tail", "30", check=False), flush=True)
        raise
    finally:
        harness.compose("down", "--volumes", "--remove-orphans", "--rmi", "local")


if __name__ == "__main__":
    main()
