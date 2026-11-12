"""CPU-only two-node Docker fault simulation for durable benchmark coordination."""

import argparse
import json
import os
import secrets
import subprocess
import tempfile
import time
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
SIMULATOR = r"""
import json,os,time
from pathlib import Path
from types import SimpleNamespace
from sqlalchemy.orm import Session
from cannery.settings import Settings
from cannery.database import make_engine
from cannery.artifacts.factory import make_store
from cannery.benchmark import coordinator
from cannery.benchmark.state import PerformanceJob,due_jobs,ensure_job
from cannery.builds.executor import ExecutionFailure
from cannery.builds.remote import RemotePending,durable_write
from cannery.builds.sandbox import WorkerProfile
s=Settings();engine=make_engine(s);store=make_store(s);root=Path('/control')
profile=WorkerProfile(target='cpu-simulation',runtime='simulation',workspace='/tmp/simulated',
 image='sha256:'+'a'*64,npu=0,device_group=0,readonly_mounts=('/usr/local/Ascend/driver',),
 compiler='simulation',fingerprint=dict(machine='aarch64',python='3.12',torch='fake',
 torch_npu='fake',cann='fake',soc='Ascend950-simulation'))
def retained(engine,job,executor):
 if (executor.remote.journal/'cleanup-done').exists():
  raise AssertionError('Cleaned input must not be read after a persisted failure')
 return {'fail':job.identity.get('fail',False)}
coordinator.retained=retained
class Executor:
 def __init__(self,claim):
  self.claim=claim;self.profile=profile;self.wheel_sha256='b'*64;self.artifacts=store
  journal=root/claim.attempt.hex;journal.mkdir(parents=True,exist_ok=True)
  self.remote=SimpleNamespace(journal=journal,call=lambda *a:{'state':'completed',
   'content':'bounded CPU simulation log'})
 def benchmark(self,attempt,fence,fail=False):
  fence();p=self.remote.journal;marker=p/'runtime-0.dispatch'
  if not marker.exists():
   with marker.open('x') as f:f.write('simulated device dispatch')
   durable_write(p/'runtime-0.json',json.dumps({'handle':'simulation:'+attempt.hex}).encode())
   durable_write(p/'dispatch-owner',os.environ['NODE'].encode())
   if not fail:raise RemotePending('Simulated lost terminal reply')
  if fail:raise ExecutionFailure('simulated_device_failure')
  if not (root/'allow-publish').exists():raise RemotePending('Simulated network disconnect')
  payload=b'{"protocol":"cpu-fault-simulation","samples_us":[1,1,1]}'
  durable_write(p/'publish-attempted',b'attempted')
  artifact=store.put('simulation/'+attempt.hex+'/samples.json',payload)
  assert store.get(artifact)==payload
  return {'passed':True,'simulation':True},artifact
 def cleanup(self,attempt):
  self.fence();durable_write(self.remote.journal/'cleanup-done',b'cleaned')
ensure_job(engine,'a'*64,'reference',{'simulation':True})
while True:
 for key in due_jobs(engine):coordinator.process(engine,key,Executor)
 time.sleep(.25)
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args()
    project = "cannery-benchmark-" + uuid4().hex[:12]
    image = project + ":app"
    env = os.environ | {"READ_TOKEN": secrets.token_hex(24), "WRITE_TOKEN": secrets.token_hex(24)}
    with tempfile.TemporaryDirectory(prefix=project) as directory:
        root = Path(directory)
        (root / "worker.py").write_text(SIMULATOR)
        overlay = root / "compose.yaml"
        overlay.write_text(f'''services:
  postgres:
    image: postgres:17
    environment:
      POSTGRES_PASSWORD: local-simulation
    healthcheck:
      test: ["CMD", "pg_isready", "-U", "postgres"]
      interval: 1s
      timeout: 2s
      retries: 30
  migrate:
    build:
      context: {ROOT}
      target: app
    image: {image}
    environment: &env
      CANNERY_DATABASE_URL: postgresql+psycopg://postgres:local-simulation@postgres/postgres
      CANNERY_ARTIFACT_BACKEND: private
      CANNERY_ARTIFACT_URL: http://artifacts:8000
      CANNERY_ARTIFACT_READ_TOKEN: ${{READ_TOKEN}}
      CANNERY_ARTIFACT_WRITE_TOKEN: ${{WRITE_TOKEN}}
    command: ["alembic", "upgrade", "head"]
    depends_on:
      postgres:
        condition: service_healthy
  artifacts:
    image: {image}
    pull_policy: never
    environment:
      <<: *env
      CANNERY_ARTIFACT_ROOT: /home/cannery/artifacts
    volumes: ["artifacts:/home/cannery/artifacts"]
    command: ["uvicorn", "cannery.artifacts.service:create_app", "--factory",
              "--host", "0.0.0.0", "--port", "8000"]
  node1: &node
    image: {image}
    pull_policy: never
    user: "0:0"
    init: true
    environment:
      <<: *env
      NODE: node1
    volumes: ["control:/control", "{root}/worker.py:/probe/worker.py:ro"]
    command: ["python", "/probe/worker.py"]
    depends_on:
      migrate:
        condition: service_completed_successfully
    pids_limit: 64
  node2:
    <<: *node
    environment:
      <<: *env
      NODE: node2
volumes:
  artifacts:
  control:
''')
        command = ["docker", "compose", "-p", project, "-f", str(overlay)]

        def compose(*values: str, timeout: int = 180) -> str:
            return subprocess.check_output(
                [*command, *values], env=env, cwd=ROOT, timeout=timeout, text=True
            )

        def query(code: str) -> dict:
            prefix = """import json
from pathlib import Path
from sqlalchemy import select
from sqlalchemy.orm import Session
from cannery.settings import Settings
from cannery.database import make_engine
from cannery.benchmark.state import PerformanceJob,PerformanceResult,ensure_job
engine=make_engine(Settings())
"""
            return json.loads(compose("exec", "-T", "node1", "python", "-c", prefix + code))

        def state(key: str) -> dict:
            return query(f"""with Session(engine) as db:
 r=db.get(PerformanceJob,{key!r})
 if r is None:
  print(json.dumps({{'status':'starting','dispatch':False,'failure_checkpoint':False,
   'cleaned':False,'pruned':False,'publish_attempted':False,'attempt':None,'attempts':0,'results':0,'error':None}}))
  raise SystemExit(0)
 p=Path('/control')/r.attempt.hex if r.attempt else None
 print(json.dumps({{'status':r.status,'attempt':r.attempt.hex if r.attempt else None,
 'attempts':r.attempts,'error':r.error,'dispatch':bool(p and (p/'runtime-0.dispatch').exists()),
 'failure_checkpoint':bool(p and (p/'benchmark-failure.json').exists()),
 'publish_attempted':bool(p and (p/'publish-attempted').exists()),
 'cleaned':bool(p and (p/'cleanup-done').exists()),
 'pruned':bool(p and (p/'cleanup-complete').exists()),
 'results':len(list(db.scalars(select(PerformanceResult).where(
 PerformanceResult.job_key==r.key))))}}))
""")

        def wait(key: str, predicate) -> dict:
            deadline = time.monotonic() + 240
            while time.monotonic() < deadline:
                row = state(key)
                if predicate(row):
                    return row
                time.sleep(0.25)
            raise RuntimeError(f"Bounded fault acceptance timed out: {row}")

        try:
            compose("build", "migrate", timeout=600)
            compose("up", "-d", "--no-build", timeout=180)
            first = wait("a" * 64, lambda r: r["dispatch"])
            compose("stop", "artifacts")
            query("Path('/control/allow-publish').touch();print('{}')")
            offline = wait(
                "a" * 64,
                lambda r: r["error"] == "artifact_storage_unavailable" and r["publish_attempted"],
            )
            assert offline["attempt"] == first["attempt"] and offline["attempts"] == 1
            assert offline["status"] == "running" and offline["results"] == 0
            compose("restart", "node1", "node2")
            assert state("a" * 64)["attempt"] == first["attempt"]
            compose("start", "artifacts")
            ready = wait("a" * 64, lambda r: r["status"] == "ready" and r["pruned"])
            assert ready["attempt"] == first["attempt"] and ready["results"] == 1
            assert ready["cleaned"] and ready["pruned"]
            query("ensure_job(engine,'a'*64,'reference',{'simulation':True});print('{}')")
            assert state("a" * 64) == ready
            compose("stop", "artifacts")
            query("ensure_job(engine,'b'*64,'reference',{'fail':True});print('{}')")
            failed = wait(
                "b" * 64,
                lambda r: (
                    r["failure_checkpoint"]
                    and r["cleaned"]
                    and r["error"] == "artifact_storage_unavailable"
                ),
            )
            assert failed["results"] == 0 and failed["status"] == "running"
            compose("restart", "node1", "node2")
            compose("start", "artifacts")
            terminal = wait("b" * 64, lambda r: r["status"] == "failed" and r["pruned"])
            assert terminal["attempt"] == failed["attempt"] and terminal["attempts"] == 1
            assert terminal["error"] == "simulated_device_failure"
            assert terminal["results"] == 1 and terminal["pruned"]
            result = {
                "simulation": "CPU only",
                "nodes": 2,
                "retained_success": ready,
                "failure_after_cleanup_outage": terminal,
            }
            if args.evidence:
                args.evidence.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result))
        except Exception:
            try:
                print(compose("logs", "--tail", "30", "node1", "node2", "artifacts"))
            except subprocess.SubprocessError:
                pass
            raise
        finally:
            compose("down", "-v", "--remove-orphans", timeout=120)
            subprocess.run(["docker", "image", "rm", image], check=False, timeout=30)


if __name__ == "__main__":
    main()
