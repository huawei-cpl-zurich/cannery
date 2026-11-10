"""Source-only intake proof using real private Git and two registrar processes."""

import json
import time
from uuid import uuid4

import registration_acceptance as harness

harness.PROJECT = f"cannery-m2-{uuid4().hex}"
harness.ENV["COMPOSE_PROJECT_NAME"] = harness.PROJECT


def revision(number: int) -> str:
    manifest = {
        "schema_version": 1,
        "kernel": "team/identity",
        "spec_version": 1,
        "implementation": {
            "device": "a5",
            "language": "catlass",
            "entrypoint": "kernel.py:forward",
            "build": {"adapter": "catlass", "inputs": ["kernel.py"], "artifact": "identity_a5"},
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
def command(*args): return subprocess.check_output(args,stderr=subprocess.DEVNULL).decode().strip()
command('git','-C',str(root),'add','candidate')
command('git','-C',str(root),'-c','user.name=fixture','-c','user.email=x@y','commit','-m','candidate{number}')
command('git','-C',str(root),'push','/fixtures/repo.git','HEAD:refs/heads/candidates')
print(command('git','-C',str(root),'rev-parse','HEAD'))
""",
    ).strip()


def submit(commit: str, parent: str | None = None) -> str:
    return harness.execute(
        "api",
        f"""
from uuid import UUID
from sqlalchemy import select
from sqlalchemy.orm import Session
from cannery.database import make_engine
from cannery.settings import Settings
from cannery.git_intake import GitSource
from cannery.registry.freeze import kernel_key
from cannery.registry.models import SpecVersion
from cannery.registry.dispatch import publish_submission
from cannery.submissions.models import RequestedTarget
from cannery.submissions.requests import new_submission,revise_submission
config=Settings(); engine=make_engine(config)
parent={parent!r}
if parent:
    key=revise_submission(engine,'team',UUID(parent),{commit!r})
else:
    with Session(engine) as session:
        spec=session.scalar(select(SpecVersion).where(SpecVersion.kernel_id==kernel_key('team','identity')))
        spec_id=spec.id
    source=GitSource(url='https://git-fixture:8443/repo.git',commit={commit!r},path='candidate')
    key=new_submission(engine,'team',spec_id,source,RequestedTarget(device='a5',language='catlass'))
publish_submission(config.redis_url.get_secret_value(),str(key)); print(key); engine.dispose()
""",
    ).strip()


def retained(identifier: str) -> None:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        status = harness.execute(
            "api",
            f"""
from uuid import UUID
from sqlalchemy.orm import Session
from cannery.database import make_engine
from cannery.settings import Settings
from cannery.submissions.state import IntakeState
engine=make_engine(Settings())
with Session(engine) as session: print(session.get(IntakeState,UUID({identifier!r})).status)
engine.dispose()
""",
        ).strip()
        if status == "awaiting_build":
            return
        assert status != "failed", identifier
        time.sleep(0.5)
    raise AssertionError("Submission intake deadline exceeded")


def main() -> None:
    try:
        harness.compose("build", timeout=600)
        harness.compose("up", "-d", "--scale", "registrar=2", "--wait")
        harness.wait_ready()
        revisions = json.loads(
            harness.compose("exec", "-T", "git-fixture", "cat", "/fixtures/revisions.json")
        )
        harness.outcome(harness.submit(revisions["v1"]))
        first = submit(revision(1))
        retained(first)
        second = submit(revision(2), first)
        retained(second)
        harness.execute(
            "api",
            f"""
from uuid import UUID
from sqlalchemy.orm import Session
from cannery.database import make_engine
from cannery.settings import Settings
from cannery.submissions.models import Candidate,Submission
engine=make_engine(Settings())
with Session(engine) as session:
    old,new=session.get(Candidate,UUID({first!r})),session.get(Candidate,UUID({second!r}))
    assert old.binding==new.binding and old.binding_id==new.binding_id==old.id
    assert old.content_sha256!=new.content_sha256
    assert session.get(Submission,new.id).parent_id==old.id
engine.dispose()
""",
        )
        print("PASS: private Git, two registrars, source-only intake, two immutable revisions")
    finally:
        harness.compose("down", "--volumes", "--remove-orphans", "--rmi", "local")


if __name__ == "__main__":
    main()
