"""Opt-in real PostgreSQL/private HTTP/two-publisher fault acceptance; CPU only."""

import json
import os
import pickle
import secrets
import shutil
import subprocess
import time
from pathlib import Path
from uuid import uuid4

import pytest
import yaml
from release_fixtures import admission as admission
from release_fixtures import arguments as arguments
from release_fixtures import benchmark as benchmark
from release_fixtures import records as records
from release_fixtures import setup_executor as setup_executor
from sqlalchemy import select

from cannery.database import Base

ROOT = Path(__file__).resolve().parents[1]
NODE = """
import json,os,pickle,sys,time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC,datetime,timedelta
from pathlib import Path
from uuid import UUID
from sqlalchemy import select,func
from sqlalchemy.orm import Session
import cannery.api
from cannery.database import Base,make_engine
from cannery.settings import Settings
from cannery.artifacts.factory import make_store
from cannery.artifacts import ArtifactError
from cannery.releases import state,worker
from cannery.releases.models import Publication,PublicationJob,PublicationEvent,Release
root=Path('/control'); engine=make_engine(Settings()); backend=make_store(Settings())
key=UUID((root/'key').read_text());actor={'namespace':'team','role':'user'}
def snapshot():
 with Session(engine) as db:
  pub=db.get(Publication,key)
  job=db.scalar(select(PublicationJob).where(PublicationJob.candidate_id==key)
   .order_by(PublicationJob.generation.desc()).limit(1))
  release=db.get(Release,pub.release_id) if pub and pub.release_id else None
  return dict(status=pub.status if pub else None,release=str(pub.release_id) if pub else None,
   job=job.status if job else None,attempts=job.attempts if job else None,
   count=db.scalar(select(func.count()).select_from(Release)),
   pin=release.manifest['sha256'] if release else None,
   activations=db.scalar(select(func.count()).select_from(PublicationEvent)
    .where(PublicationEvent.kind=='published')))
mode=sys.argv[1] if len(sys.argv)>1 else 'worker'
if mode=='seed':
 # Only our locally generated trusted fixture pickle is mounted into these isolated containers.
 tables,blobs=pickle.loads((root/'fixture').read_bytes())
 with engine.begin() as c:
  for name,rows in tables:
   if rows:c.execute(Base.metadata.tables[name].insert(),rows)
 for sha,raw in blobs.items():assert backend.put('fixture/'+sha,raw).sha256==sha
 with ThreadPoolExecutor(max_workers=4) as pool:
  jobs=list(pool.map(lambda _:state.request_publication(engine,key,actor),range(4)))
 assert len(set(jobs))==1
 print(json.dumps(snapshot()));raise SystemExit
if mode=='reuse':
 from cannery.git_intake import GitSource,SnapshotBinding
 from cannery.git_intake.snapshots import SnapshotStore,SelectedTree
 from cannery.submissions.models import Candidate,Submission,SubmissionReuse
 from cannery.submissions.state import IntakeState
 from cannery.submissions.reuse import reuse_completed
 from cannery.submissions.api import get_info
 from cannery.builds.state import BuildState
 snapshots=SnapshotStore(root/'snapshots')
 with Session(engine) as db,db.begin():
  old=db.get(Candidate,key);parent=db.get(Submission,key)
  source=GitSource.model_validate(parent.source|{'commit':'b'*40})
  selected=snapshots.read(SnapshotBinding.model_validate(old.snapshot))
  binding=snapshots.commit(old.snapshot['internal_repository'],source,
   SelectedTree(selected.files,selected.modes,'c'*40))
  sub=Submission(namespace=parent.namespace,spec_id=parent.spec_id,
   source=source.model_dump(mode='json'),target=parent.target)
  db.add(sub);db.flush()
  candidate=Candidate(id=sub.id,binding_id=sub.id,binding=old.binding,manifest=old.manifest,
   snapshot=binding.model_dump(mode='json'),content_sha256=old.content_sha256,
   file_sha256=old.file_sha256)
  db.add_all([candidate,IntakeState(id=sub.id,status='awaiting_build')]);db.flush()
  reuse_completed(engine,db,candidate,snapshots,lambda:backend);db.flush()
  assert db.get(SubmissionReuse,sub.id).original_id==key
  assert db.get(BuildState,sub.id) is None
  info=get_info(db,'team',sub.id)
  assert info['status']=='complete' and info['reused_from']==str(key)
 from sqlalchemy.exc import DBAPIError
 try:
  with engine.begin() as c:c.exec_driver_sql('UPDATE submission_reuse SET original_id=original_id')
 except DBAPIError:pass
 else:raise AssertionError('PostgreSQL reuse trigger did not protect immutable links')
 print(json.dumps({'reused':True,'new_builds':0}));raise SystemExit
if mode=='query':
 command=sys.argv[2]
 if command=='expire':
  with Session(engine) as db,db.begin():
   for job in db.scalars(select(PublicationJob).where(PublicationJob.status=='running')):
    job.lease_until=datetime.now(UTC)-timedelta(seconds=1)
 if command=='withdraw':state.withdraw(engine,key,actor)
 if command=='publish':state.request_publication(engine,key,actor)
 print(json.dumps(snapshot()));raise SystemExit
class Store:
 def get(self,ref):
  try:return backend.get(ref)
  except ArtifactError:
   (root/'storage-failed').touch()
   raise
 def put(self,path,raw):
  ref=backend.put(path,raw)
  if path.endswith('/manifest.json'):
   (root/'owner').write_text(os.environ['NODE'])
   deadline=time.monotonic()+90
   while not (root/'allow').exists():
    if time.monotonic()>deadline:raise RuntimeError('Bounded upload gate timed out')
    time.sleep(.1)
  return ref
while True:
 if (root/'storage-failed').exists():
  time.sleep(.1);continue
 for identifier in state.due_jobs(engine):worker.process(engine,identifier,Store())
 time.sleep(.2)
"""


@pytest.mark.skipif(
    os.environ.get("CANNERY_RELEASE_DOCKER") != "1", reason="opt-in Docker acceptance"
)
def test_two_publishers_recover_and_fence_withdrawal(admission, setup_executor, tmp_path):
    engine, key, memory, _ = admission
    project = "cannery-release-" + uuid4().hex[:10]
    image = project + ":app"
    env = os.environ | {"READ_TOKEN": secrets.token_hex(24), "WRITE_TOKEN": secrets.token_hex(24)}
    with engine.connect() as connection:
        tables = [
            (t.name, [dict(r._mapping) for r in connection.execute(select(t))])
            for t in Base.metadata.sorted_tables
        ]
    control = tmp_path / "control"
    control.mkdir()
    (control / "fixture").write_bytes(pickle.dumps((tables, memory.data)))
    (control / "key").write_text(str(key))
    shutil.copytree(setup_executor[0].snapshots.root, control / "snapshots")
    node = tmp_path / "node.py"
    node.write_text(NODE)
    settings = {
        "CANNERY_DATABASE_URL": "postgresql+psycopg://postgres:local@postgres/postgres",
        "CANNERY_ARTIFACT_BACKEND": "private",
        "CANNERY_ARTIFACT_URL": "http://artifacts:8000",
        "CANNERY_ARTIFACT_READ_TOKEN": "${READ_TOKEN}",
        "CANNERY_ARTIFACT_WRITE_TOKEN": "${WRITE_TOKEN}",
    }
    worker = {
        "image": image,
        "pull_policy": "never",
        "user": "0:0",
        "init": True,
        "environment": settings | {"NODE": "node1"},
        "volumes": [f"{control}:/control", f"{node}:/probe/node.py:ro"],
        "command": ["python", "/probe/node.py"],
        "pids_limit": 64,
    }
    services = {
        "postgres": {
            "image": "postgres:17",
            "environment": {"POSTGRES_PASSWORD": "local"},
            "healthcheck": {
                "test": ["CMD", "pg_isready", "-U", "postgres"],
                "interval": "1s",
                "timeout": "2s",
                "retries": 30,
            },
        },
        "migrate": {
            "build": {"context": str(ROOT), "target": "app"},
            "image": image,
            "environment": settings,
            "command": ["alembic", "upgrade", "head"],
            "depends_on": {"postgres": {"condition": "service_healthy"}},
        },
        "artifacts": {
            "image": image,
            "pull_policy": "never",
            "environment": settings | {"CANNERY_ARTIFACT_ROOT": "/home/cannery/artifacts"},
            "volumes": ["artifacts:/home/cannery/artifacts"],
            "command": [
                "uvicorn",
                "cannery.artifacts.service:create_app",
                "--factory",
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
            ],
        },
        "node1": worker,
        "node2": worker | {"environment": settings | {"NODE": "node2"}},
    }
    # Delayed startup makes resuming before HTTP readiness a deterministic fault.
    services["artifacts"]["command"] = [
        "python",
        "-c",
        "import os,time; time.sleep(2); os.execvp('uvicorn', "
        "['uvicorn','cannery.artifacts.service:create_app','--factory',"
        "'--host','0.0.0.0','--port','8000'])",
    ]
    for service in services.values():
        service["labels"] = {"cannery.task": project}
    overlay = tmp_path / "compose.yaml"
    overlay.write_text(yaml.safe_dump({"services": services, "volumes": {"artifacts": {}}}))
    command = ["docker", "compose", "-p", project, "-f", str(overlay)]

    def compose(*args, timeout=120):
        return subprocess.check_output(command + list(args), env=env, timeout=timeout, text=True)

    def query(command="status"):
        return json.loads(
            compose(
                "run", "--rm", "--no-deps", "node1", "python", "/probe/node.py", "query", command
            )
        )

    def wait(predicate):
        deadline = time.monotonic() + 120
        row = None
        while time.monotonic() < deadline:
            row = query()
            if predicate(row):
                return row
            time.sleep(0.3)
        raise AssertionError(f"Publication acceptance deadline: {row}")

    def artifact_ready():
        compose(
            "run",
            "--rm",
            "--no-deps",
            "node1",
            "python",
            "-c",
            "import time,httpx\nfor i in range(30):\n try:\n"
            '  httpx.get("http://artifacts:8000/health/live",timeout=1);break\n'
            " except httpx.HTTPError:time.sleep(.2)\n"
            "else:raise RuntimeError('Artifact HTTP readiness deadline exceeded')",
        )

    built = False
    try:
        compose("build", "migrate", timeout=600)
        built = True
        compose("up", "-d", "postgres", "artifacts", "migrate")
        compose("wait", "migrate")
        artifact_ready()
        compose("run", "--rm", "--no-deps", "node1", "python", "/probe/node.py", "seed")
        assert json.loads(
            compose("run", "--rm", "--no-deps", "node1", "python", "/probe/node.py", "reuse")
        )["reused"]
        compose("up", "-d", "--no-deps", "node1", "node2")
        first = wait(lambda r: (control / "owner").exists())
        assert first["status"] == "publishing" and first["count"] == 0
        owner = (control / "owner").read_text()
        compose("kill", "-s", "SIGKILL", owner)
        query("expire")
        (control / "allow").touch()
        ready = wait(lambda r: r["status"] == "published")
        assert ready["count"] == 1 and ready["attempts"] == 1 and ready["activations"] == 1
        assert query("publish") == ready
        query("withdraw")
        (control / "allow").unlink()
        (control / "owner").unlink()
        compose("stop", "artifacts")
        with pytest.raises(subprocess.CalledProcessError):
            compose("run", "--rm", "--no-deps", "node1", "python", "/probe/node.py", "reuse")
        query("publish")
        offline = wait(lambda r: r["job"] == "queued" and r["attempts"] >= 1)
        assert offline["status"] == "publishing" and offline["count"] == 1
        compose("start", "artifacts")
        artifact_ready()
        assert json.loads(
            compose("run", "--rm", "--no-deps", "node1", "python", "/probe/node.py", "reuse")
        )["reused"]
        (control / "storage-failed").unlink(missing_ok=True)
        wait(lambda r: (control / "owner").exists())
        query("withdraw")
        (control / "allow").touch()
        fenced = wait(lambda r: r["job"] == "cancelled")
        assert fenced["status"] == "withdrawn" and fenced["activations"] == 1
        query("publish")
        restored = wait(lambda r: r["status"] == "published")
        assert restored["release"] == ready["release"] and restored["pin"] == ready["pin"]
        assert restored["count"] == 1 and restored["activations"] == 2
        print(
            json.dumps(
                {
                    "cpu_simulation": True,
                    "nodes": 2,
                    "deduplicated": True,
                    "storage_recovery": True,
                    "stale_owner_fenced": True,
                    "withdrawal_fenced": True,
                    "republish_same_pin": True,
                }
            )
        )
    finally:
        try:
            if built:
                compose(
                    "run",
                    "--rm",
                    "--no-deps",
                    "node1",
                    "chown",
                    "-R",
                    f"{os.getuid()}:{os.getgid()}",
                    "/control",
                    timeout=30,
                )
        finally:
            compose("down", "-v", "--remove-orphans")
            subprocess.run(["docker", "image", "rm", image], check=False, timeout=30)
