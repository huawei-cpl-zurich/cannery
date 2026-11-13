"""Actual offline PG/Git/blob/journal round trip and duplicate delivery; CPU only."""

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from uuid import uuid4

import pytest
import yaml

from cannery.maintenance.capture import Docker, capture, inventory
from cannery.maintenance.restore import restore

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(
    os.environ.get("CANNERY_BACKUP_DOCKER") != "1", reason="opt-in Docker restore drill"
)
def test_offline_backup_restore_and_durable_replay(tmp_path):
    source_name = "cannery-m7a-source-" + uuid4().hex[:8]
    target_name = "cannery-m7a-target-" + uuid4().hex[:8]
    image = source_name + ":app"
    roles = {
        "api": ("git", "/var/lib/cannery/snapshots"),
        "artifacts": ("blobs", "/home/cannery/artifacts"),
        "builder": ("builds", "/var/lib/cannery/builds"),
        "registrar": ("baseline", "/var/lib/cannery/baselines"),
    }
    services = {
        "postgres": {
            "image": "postgres:17",
            "environment": {"POSTGRES_PASSWORD": "local"},
            "volumes": ["pg:/var/lib/postgresql/data"],
        }
    }
    for service, (volume, path) in roles.items():
        services[service] = {
            "image": image,
            "environment": {
                "CANNERY_DATABASE_URL": "postgresql+psycopg://postgres:local@postgres/postgres",
                "CANNERY_BASELINE_ROOT": "/var/lib/cannery/baselines",
            },
            "volumes": [f"{volume}:{path}"],
        }
    configuration = tmp_path / "compose.yaml"
    configuration.write_text(
        yaml.safe_dump(
            {
                "services": services,
                "volumes": {v: {} for v in ["pg", "git", "blobs", "builds", "baseline"]},
            }
        )
    )

    def compose(project, *args):
        return subprocess.check_output(
            ["docker", "compose", "-p", project, "-f", str(configuration), *args],
            text=True,
            timeout=120,
        )

    docker = Docker()
    try:
        subprocess.run(
            ["docker", "build", "--target", "app", "-t", image, str(ROOT)], check=True, timeout=600
        )
        for project in [source_name, target_name]:
            compose(project, "create", "--no-build")
            compose(project, "up", "-d", "postgres")
            deadline = time.monotonic() + 60
            while True:
                try:
                    inventory(docker, project).sql("SELECT 1")
                    break
                except ValueError:
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(0.2)
        compose(source_name, "run", "--rm", "--no-deps", "api", "alembic", "upgrade", "head")
        source = inventory(docker, source_name)
        probe = uuid4()
        source.sql(f"INSERT INTO probe_jobs VALUES ('{probe}',now(),'queued',0)")
        blob = b"immutable cached outputs"
        digest = hashlib.sha256(blob).hexdigest()
        source.sql("CREATE TABLE maintenance_fixture_refs (data jsonb)")
        reference = json.dumps(
            dict(repository="private", revision=digest, path="outputs.tar", sha256=digest)
        )
        source.sql(f"INSERT INTO maintenance_fixture_refs VALUES ('{reference}')")
        source.docker.volume(
            image,
            source.volumes["artifacts"],
            f"from pathlib import Path; Path('/data/{digest}').write_bytes({blob!r})",
            writable=True,
        )
        seed_git = """import os,subprocess
from pathlib import Path
repo='/data/snapshot.git'
subprocess.run(['git','init','--bare',repo],check=True,stdout=subprocess.DEVNULL)
def git(*args,data=None):
 return subprocess.check_output(['git','-C',repo,*args],input=data).strip()
blob=git('hash-object','-w','--stdin',data=b'seeded-source')
tree=git('mktree',data=b'100644 blob '+blob+b'\\tspec.py\\n')
os.environ.update(GIT_AUTHOR_NAME='fixture',GIT_COMMITTER_NAME='fixture',
                  GIT_AUTHOR_EMAIL='fixture@local',GIT_COMMITTER_EMAIL='fixture@local')
commit=git('commit-tree',tree.decode(),'-m','fixture')
git('update-ref','refs/heads/main',commit.decode())
"""
        source.docker.volume(image, source.volumes["snapshots"], seed_git, writable=True)
        for role in ["build_control", "baseline_control"]:
            source.docker.volume(
                image,
                source.volumes[role],
                """import os
from pathlib import Path
root=Path('/data/attempt');root.mkdir(mode=0o700)
file=root/'receipt.json';file.write_text('{"handle":"remote:fixture:job:original"}')
file.chmod(0o600);os.chown(file,12345,12345);os.chown(root,12345,12345)
""",
                writable=True,
            )
        saved = capture(docker, source_name, tmp_path / "backup")
        assert len(saved.files) == 5
        restored = restore(docker, tmp_path / "backup", target_name)
        assert restored == saved
        target = inventory(docker, target_name)
        assert (
            target.sql(f"SELECT status||':'||effects FROM probe_jobs WHERE id='{probe}'")
            == "queued:0"
        )
        assert json.loads(target.sql("SELECT data FROM maintenance_fixture_refs")) == json.loads(
            reference
        )
        target.docker.volume(
            image,
            target.volumes["artifacts"],
            f"from pathlib import Path; assert Path('/data/{digest}').read_bytes()=={blob!r}",
        )
        target.docker.volume(
            image,
            target.volumes["snapshots"],
            """import subprocess
repo='/data/snapshot.git'
subprocess.run(['git','-C',repo,'fsck','--strict'],check=True)
assert subprocess.check_output(['git','-C',repo,'show','main:spec.py'])==b'seeded-source'
""",
        )
        for role in ["build_control", "baseline_control"]:
            target.docker.volume(
                image,
                target.volumes[role],
                """from pathlib import Path
root=Path('/data/attempt');file=root/'receipt.json'
assert root.stat().st_uid==12345 and file.stat().st_mode & 0o777==0o600
assert file.read_text()=='{"handle":"remote:fixture:job:original"}'
""",
            )
        # A restored SQL queue needs no Redis backup. Two independent deliveries have one effect.
        replay = (
            "from cannery.database import make_engine; from cannery.settings import Settings; "
            "from cannery.jobs import pending_probes,complete_probe; "
            f"engine=make_engine(Settings()); assert '{probe}' in pending_probes(engine); "
            f"complete_probe(engine,'{probe}'); complete_probe(engine,'{probe}')"
        )
        compose(target_name, "run", "--rm", "--no-deps", "api", "python", "-c", replay)
        assert (
            target.sql(f"SELECT status||':'||effects FROM probe_jobs WHERE id='{probe}'")
            == "complete:1"
        )
        with pytest.raises(ValueError, match="must be empty"):
            restore(docker, tmp_path / "backup", target_name)
        print(
            json.dumps(
                {
                    "roundtrip": True,
                    "redis_restored": False,
                    "duplicate_effects": 1,
                    "journals_preserved": True,
                    "workers_started_by_restore": False,
                }
            )
        )
    finally:
        for project in [source_name, target_name]:
            compose(project, "down", "-v", "--remove-orphans")
        subprocess.run(["docker", "image", "rm", image], check=False, timeout=30)
