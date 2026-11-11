"""Two Docker registrars: seeded ingress, private storage faults and restart reuse."""

import argparse
import json
import os
import secrets
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.request import Request, urlopen
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "https://github.com/huawei-cpl-zurich/helloworld-kernel.git"
COMMIT = "c9874e240e9c0f59b90805e06168a3701457d9a4"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args()
    project = "cannery-numerics-" + uuid4().hex[:12]
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    env = {
        **os.environ,
        "COMPOSE_PROJECT_NAME": project,
        "CANNERY_PORT": str(port),
        "CANNERY_DEV_IDENTITY_ENABLED": "true",
        "CANNERY_DEV_ROLE": "admin",
        "CANNERY_ARTIFACT_READ_TOKEN": secrets.token_hex(32),
        "CANNERY_ARTIFACT_WRITE_TOKEN": secrets.token_hex(32),
        "CANNERY_GIT_REMOTES": json.dumps([{"url": REPOSITORY, "namespaces": ["team"]}]),
    }
    owners = set()
    with tempfile.TemporaryDirectory(prefix="cannery-numerics-") as directory:
        overlay = Path(directory) / "numerics.yaml"
        overlay.write_text("""services:
  registrar:
    environment:
      CANNERY_BASELINE_ROOT: /var/lib/cannery/baselines
      CANNERY_BASELINE_VOLUME: ${COMPOSE_PROJECT_NAME}-baseline-control
      CANNERY_ARTIFACT_BACKEND: private
      CANNERY_ARTIFACT_URL: http://artifacts:8000
      CANNERY_ARTIFACT_READ_TOKEN: ${CANNERY_ARTIFACT_READ_TOKEN}
      CANNERY_ARTIFACT_WRITE_TOKEN: ${CANNERY_ARTIFACT_WRITE_TOKEN}
    volumes:
      - baseline-control:/var/lib/cannery/baselines
volumes:
  baseline-control:
    name: ${COMPOSE_PROJECT_NAME}-baseline-control
""")
        command = [
            "docker",
            "compose",
            "-p",
            project,
            "-f",
            "compose.yaml",
            "-f",
            "compose.artifacts.yaml",
            "-f",
            str(overlay),
        ]

        def compose(*values: str, timeout: int = 180) -> str:
            return subprocess.check_output(
                [*command, *values],
                cwd=ROOT,
                env=env,
                timeout=timeout,
                stderr=subprocess.PIPE,
                text=True,
            )

        def request(path: str, body: dict | None = None) -> dict:
            with urlopen(
                Request(
                    f"http://127.0.0.1:{port}" + path,
                    data=None if body is None else json.dumps(body).encode(),
                    headers={"Content-Type": "application/json"},
                ),
                timeout=8,
            ) as response:
                return json.load(response)

        def register(path: str) -> dict:
            return request(
                "/api/v1/kernels",
                {
                    "name": "helloworld",
                    "source": {"url": REPOSITORY, "commit": COMMIT, "path": path},
                },
            )

        def wait_job(job: dict) -> dict:
            for _ in range(120):
                result = request(job["status_url"])
                if result["status"] in ("complete", "failed"):
                    assert result["status"] == "complete", result
                    return result
                time.sleep(2)
            raise RuntimeError("Registration exceeded bounded acceptance deadline")

        def snapshot() -> list[dict]:
            code = """import json
from pathlib import Path
from sqlalchemy.orm import Session
from sqlalchemy import select
from cannery.settings import Settings
from cannery.database import make_engine
from cannery.numerics.baseline_state import BaselineState
s=Settings()
with Session(make_engine(s)) as db:
    result=[]
    for row in db.scalars(select(BaselineState)):
        root=s.baseline_root/row.key/(row.attempt.hex if row.attempt else 'none')
        result.append({'key':row.key,'status':row.status,'attempts':row.attempts,
          'attempt':row.attempt.hex if row.attempt else None,'artifact':row.artifact,
          'checkpoint':(root/'output/result.json').exists(),
          'dispatches':len(list((s.baseline_root/row.key).glob('*/create.dispatch'))),
          'raw_outputs':len(list((s.baseline_root/row.key).glob('*/output/result.json')))})
    print(json.dumps(result))
"""
            values = json.loads(compose("exec", "-T", "registrar", "python", "-c", code))
            owners.update(row["attempt"] for row in values if row["attempt"])
            return values

        artifacts = ""
        detached = False
        try:
            print("Building isolated CPU ingress services", flush=True)
            compose(
                "up",
                "-d",
                "--build",
                "--wait",
                "--scale",
                "registrar=2",
                "api",
                "registrar",
                "reconciler",
                "artifacts",
                timeout=300,
            )
            original = wait_job(register("spec"))
            artifacts = compose("ps", "-q", "artifacts").strip()
            network = project + "_default"
            subprocess.run(
                ["docker", "network", "disconnect", network, artifacts],
                check=True,
                timeout=20,
                capture_output=True,
            )
            detached = True
            first = register("spec-v2")
            before = None
            for _ in range(60):
                rows = snapshot()
                if rows and rows[0]["checkpoint"]:
                    before = rows[0]
                    break
                time.sleep(2)
            assert before and before["status"] == "generating", before
            assert before["dispatches"] == before["attempts"] == 1
            compose("restart", "registrar", timeout=45)
            subprocess.run(
                ["docker", "network", "connect", "--alias", "artifacts", network, artifacts],
                check=True,
                timeout=20,
                capture_output=True,
            )
            detached = False
            completed = wait_job(first)
            after = snapshot()[0]
            assert after["status"] == "ready" and after["attempt"] == before["attempt"]
            assert after["attempts"] == after["dispatches"] == 1 and after["raw_outputs"] == 0
            # A ready cache with unavailable storage must wait, never rerun the oracle.
            subprocess.run(
                ["docker", "network", "disconnect", network, artifacts],
                check=True,
                timeout=20,
                capture_output=True,
            )
            detached = True
            repeat = register("spec-v2")
            time.sleep(8)
            assert request(repeat["status_url"])["status"] in ("queued", "running")
            assert snapshot()[0] == after
            subprocess.run(
                ["docker", "network", "connect", "--alias", "artifacts", network, artifacts],
                check=True,
                timeout=20,
                capture_output=True,
            )
            detached = False
            repeated = wait_job(repeat)
            assert repeated["spec"]["id"] == completed["spec"]["id"]
            assert snapshot()[0] == after
            result = {
                "project": project,
                "source": COMMIT,
                "v1_id": original["spec"]["id"],
                "v2_id": completed["spec"]["id"],
                "baseline": after,
                "two_registrars": True,
                "storage_disconnect_recovered": True,
                "coordinator_restart_same_attempt": True,
                "repeat_reused_outputs": True,
            }
            if args.evidence:
                args.evidence.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result), flush=True)
        finally:
            if detached:
                subprocess.run(
                    [
                        "docker",
                        "network",
                        "connect",
                        "--alias",
                        "artifacts",
                        project + "_default",
                        artifacts,
                    ],
                    timeout=20,
                    capture_output=True,
                )
            for owner in owners:
                name = "cannery-oracle-" + owner
                info = subprocess.run(
                    ["docker", "inspect", name], capture_output=True, text=True, timeout=20
                )
                if (
                    info.returncode == 0
                    and json.loads(info.stdout)[0]["Config"]["Labels"].get("cannery.oracle.owner")
                    == owner
                ):
                    subprocess.run(["docker", "rm", "-f", name], timeout=20, capture_output=True)
            compose("down", "-v", "--remove-orphans", timeout=90)


if __name__ == "__main__":
    main()
