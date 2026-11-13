"""Two real SQL worker nodes, device claims and a deterministic network partition."""

import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import pytest

ROOT = Path(__file__).resolve().parents[1]
COMMON = """
import json,os,time
from pathlib import Path
from uuid import UUID
from datetime import UTC,datetime,timedelta
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from cannery.api import create_app
from cannery.database import Base
from cannery.builds.state import (
 BuildState,BuildClaim,claim_build,heartbeat as build_heartbeat,finish_build
)
from cannery.numerics.state import (
 NumericalState,NumericalClaim,claim_numerical,heartbeat,finish_numerical
)
from cannery.benchmark.state import PerformanceJob,PerformanceClaim,ensure_job,claim_job,finish_job
from cannery.submissions.models import Submission,Candidate
from cannery.registry.models import Kernel,SpecVersion
from cannery.artifacts import ArtifactRef
engine=create_engine(os.environ['TEST_DATABASE'],connect_args={'connect_timeout':2,'tcp_user_timeout':2000})
artifact=ArtifactRef(repository='private',revision='a'*64,sha256='a'*64,path='fixture.bin')
"""
SEED = """
Base.metadata.create_all(engine)
ids={'a3':str(UUID(int=3)),'a5':str(UUID(int=5))}
with Session(engine) as db,db.begin():
 db.add(Kernel(id='kernel',namespace='team',name='fixture'));db.flush()
 db.add(SpecVersion(id=UUID(int=1),kernel_id='kernel',version=1,content_sha256='a'*64,manifest={},snapshot={},file_sha256={},smoke={}));db.flush()
 for device,key in ids.items():
  db.add(Submission(id=UUID(key),namespace='team',spec_id=UUID(int=1),source={},target={'device':device,'language':'ascendc'}));db.flush()
  db.add(Candidate(id=UUID(key),binding_id=UUID(key),binding={},manifest={'implementation':{'device':device}},snapshot={},content_sha256='a'*64,file_sha256={}));db.flush()
  db.add(BuildState(id=UUID(key)));db.add(NumericalState(id=UUID(key)))
for device in ids:
 ensure_job(engine,device[-1]*64,'reference',{'cohort':{'device':device}})
"""
CLAIM = """
device=os.environ['TEST_DEVICE'];other='a5' if device=='a3' else 'a3'
key=UUID(int=3 if device=='a3' else 5);foreign=UUID(int=5 if device=='a3' else 3)
assert claim_build(engine,foreign,device=device) is None
assert claim_numerical(engine,foreign,device=device) is None
assert claim_job(engine,other[-1]*64,device=device) is None
build=claim_build(engine,key,device=device);numeric=claim_numerical(engine,key,device=device)
performance=claim_job(engine,device[-1]*64,device=device)
assert build and numeric and performance
Path('/tmp/claims.json').write_text(json.dumps({'build':[str(build.id),str(build.attempt),str(build.token)],'numeric':[str(numeric.id),str(numeric.attempt),str(numeric.token)],'performance':[performance.key,str(performance.attempt),str(performance.token)]}))
"""
LOAD = """
saved=json.loads(Path('/tmp/claims.json').read_text())
build=BuildClaim(*(UUID(v) for v in saved['build']))
numeric=NumericalClaim(*(UUID(v) for v in saved['numeric']))
values=saved['performance'];performance=PerformanceClaim(values[0],UUID(values[1]),UUID(values[2]))
"""
FINISH = """
assert finish_build(engine,build,artifact=artifact)
assert finish_numerical(engine,numeric,report={'passed':True})
assert finish_job(engine,performance,report={'passed':True},artifact=artifact)
"""


@pytest.mark.skipif(
    os.environ.get("CANNERY_DEVICE_DOCKER") != "1", reason="opt-in dual-device Docker fault drill"
)
def test_dual_device_workers_recover_partition_without_cross_claim_or_duplicate_attempt():
    prefix = "cannery-m7b-routing-" + uuid4().hex[:8]
    network, postgres, image = prefix + "-net", prefix + "-pg", prefix + ":app"
    nodes = {device: prefix + "-" + device for device in ["a3", "a5"]}

    def docker(*args, source=None, timeout=30, check=True):
        return subprocess.run(
            ["docker", *args],
            input=source,
            capture_output=True,
            text=True,
            check=check,
            timeout=timeout,
        ).stdout

    def execute(device, code):
        return docker("exec", "-i", nodes[device], "python", "-", source=COMMON + code, timeout=15)

    try:
        docker("build", "--target", "app", "-t", image, str(ROOT), timeout=600)
        docker("network", "create", "--label", "cannery.task=m7b-routing", network)
        docker(
            "run",
            "-d",
            "--name",
            postgres,
            "--label",
            "cannery.task=m7b-routing",
            "--network",
            network,
            "-e",
            "POSTGRES_PASSWORD=local",
            "postgres:17",
        )
        deadline = time.monotonic() + 40
        while "accepting connections" not in docker(
            "exec", postgres, "pg_isready", "-U", "postgres", check=False
        ):
            assert time.monotonic() < deadline
            time.sleep(0.2)
        for device, name in nodes.items():
            docker(
                "run",
                "-d",
                "--name",
                name,
                "--label",
                "cannery.task=m7b-routing",
                "--network",
                network,
                "--read-only",
                "--tmpfs",
                "/tmp:size=16m",
                "--pids-limit",
                "64",
                "-e",
                f"TEST_DATABASE=postgresql+psycopg://postgres:local@{postgres}/postgres",
                "-e",
                f"TEST_DEVICE={device}",
                image,
                "python",
                "-c",
                "import time;time.sleep(600)",
            )
        execute("a5", SEED)
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda device: execute(device, CLAIM), nodes))
        docker("network", "disconnect", network, nodes["a3"])
        execute(
            "a3",
            LOAD
            + """
started=time.monotonic()
try:heartbeat(engine,numeric)
except SQLAlchemyError:pass
else:raise AssertionError('partition did not interrupt the SQL worker')
assert time.monotonic()-started<6
""",
        )
        execute(
            "a5",
            LOAD
            + FINISH
            + """
with Session(engine) as db,db.begin():
 for model in [BuildState,NumericalState]:
  row=db.get(model,UUID(int=3));row.lease_until=datetime.now(UTC)-timedelta(seconds=1)
 row=db.get(PerformanceJob,'3'*64);row.lease_until=row.due_at=datetime.now(UTC)-timedelta(seconds=1)
""",
        )
        docker("network", "connect", network, nodes["a3"])
        execute(
            "a3",
            LOAD
            + """
assert not heartbeat(engine,numeric) and not build_heartbeat(engine,build)
new_build=claim_build(engine,build.id,device='a3')
new_numeric=claim_numerical(engine,numeric.id,device='a3')
new_performance=claim_job(engine,performance.key,device='a3')
assert new_build.attempt==build.attempt and new_numeric.attempt==numeric.attempt
assert new_performance.attempt==performance.attempt
assert not finish_numerical(engine,numeric,report={'passed':True})
build,numeric,performance=new_build,new_numeric,new_performance
"""
            + FINISH
            + """
with Session(engine) as db:
 for key in [UUID(int=3),UUID(int=5)]:
  assert db.get(BuildState,key).attempts==db.get(NumericalState,key).attempts==1
 for key in ['3'*64,'5'*64]:assert db.get(PerformanceJob,key).attempts==1
""",
        )
    finally:
        for name in [*nodes.values(), postgres]:
            docker("rm", "-f", "-v", name, check=False)
        docker("network", "rm", network, check=False)
        docker("image", "rm", image, check=False)
