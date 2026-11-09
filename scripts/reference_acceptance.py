"""CPU reference sandbox checks with task-owned image/container cleanup."""

import json
import os
import subprocess
from pathlib import Path

from cannery.registry.manifest import validate_spec
from cannery.registry.sandbox import SmokeError, new_handle, run_smoke, split_handle

ROOT = Path(__file__).resolve().parents[1]
IMAGE = f"cannery-reference-check-{os.getpid()}"


def files(
    reference: bytes, cases: bytes | None = None, torch: str = "2.6.0+cpu"
) -> dict[str, bytes]:
    manifest = {
        "schema_version": 1,
        "kernel": {"namespace": "sandbox", "name": "identity"},
        "spec": {
            "version": 1,
            "reference": {"file": "reference.py", "function": "reference"},
            "cases": {"file": "cases.py", "function": "generate_cases"},
            "interface": {"inputs": ["x"], "outputs": ["result"]},
            "validation": {"metric": "torch_close", "default_rtol": 0.001, "default_atol": 0.0001},
        },
    }
    return {
        "spec/cannery.yaml": json.dumps(manifest).encode(),
        "spec/reference.py": reference,
        "spec/cases.py": cases
        if cases is not None
        else b"import torch\ndef generate_cases(): yield {'x': torch.ones(2)}\n",
        "spec/environment.lock": json.dumps(
            {"schema_version": 1, "python": "3.12", "torch": torch}
        ).encode(),
    }


def main() -> None:
    try:
        subprocess.run(
            ["docker", "build", "-f", "Dockerfile.smoke", "-t", IMAGE, "."],
            cwd=ROOT,
            check=True,
            timeout=300,
        )
        data = files(b"def reference(x): return x\n")
        spec = validate_spec(data, "spec")
        evidence = run_smoke(spec, data, "spec", image=IMAGE)
        assert evidence.cases == 1 and evidence.spec_sha256 == spec.content_sha256
        isolated = files(b"""
import os, socket
from pathlib import Path

def reference(x):
    assert 'CANNERY_DATABASE_URL' not in os.environ
    assert 'CANNERY_HF_TOKEN' not in os.environ
    assert os.getuid() == 10001
    assert not Path('/var/run/docker.sock').exists()
    try:
        fd = os.open('/proc/1/fd/1', os.O_WRONLY)
    except PermissionError:
        pass
    else:
        os.close(fd)
        raise AssertionError('supervisor stdout is accessible')
    try:
        Path('/etc/cannery-escape').write_text('bad')
    except OSError:
        pass
    else:
        raise AssertionError('root filesystem is writable')
    with socket.socket() as client:
        client.settimeout(.2)
        try:
            client.connect(('1.1.1.1', 443))
        except OSError:
            pass
        else:
            raise AssertionError('network is available')
    return x
""")
        run_smoke(validate_spec(isolated, "spec"), isolated, "spec", image=IMAGE)
        adaptive = files(
            b"""import __main__, json, os, sys

def send(request, **fields):
    os.write(1, (json.dumps(dict(nonce=request['nonce'], **fields)) + '\\n').encode())

send(__main__.request, ready=True)
for line in sys.stdin:
    request = json.loads(line)
    if request['op'] == 'next':
        send(request, done=False, inputs=['x'])
    else:
        send(request, outputs=[dict(shape=[2], dtype='torch.float32')])
""",
            cases=b"def generate_cases(): return []",
        )
        observation = run_smoke(validate_spec(adaptive, "spec"), adaptive, "spec", image=IMAGE)
        assert observation.assurance == "diagnostic"  # Never proof of callable execution.
        # A colliding name belongs to another invocation; failed launch must
        # preserve it even when it has the same public name prefix.
        ownership = new_handle()
        name, _ = split_handle(ownership)
        foreign = subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                name,
                "--label",
                "cannery.smoke.owner=foreign",
                IMAGE,
                "python",
                "-c",
                "import time; time.sleep(60)",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        try:
            try:
                run_smoke(
                    spec,
                    files(b"def reference(x): return x\n"),
                    "spec",
                    image=IMAGE,
                    handle=ownership,
                )
            except SmokeError as error:
                assert error.handle == ownership
            else:
                raise AssertionError("Name collision unexpectedly succeeded")
            state = subprocess.run(
                ["docker", "inspect", "--format", "{{.State.Running}}", foreign],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            assert state.stdout.strip() == "true"
        finally:
            subprocess.run(["docker", "rm", "-f", foreign], check=True, timeout=5)
        bad = [
            files(
                b"""import os
os.write(1, b'{"python":"3.12","torch":"2.6.0+cpu","cases":3}\\n')
os._exit(0)
""",
                cases=b"def generate_cases(): return []",
            ),
            files(b"def reference(x): return x", cases=b"def generate_cases(): return []"),
            files(b"def reference(x): return x", torch="0.0.0"),
            files(b"def reference(x):\n while True: pass\n"),
            files(b"import os\ndef reference(x):\n os.write(1, b'x' * 2097152)\n return x"),
        ]
        for data in bad:
            try:
                run_smoke(validate_spec(data, "spec"), data, "spec", image=IMAGE, timeout=4)
            except SmokeError:
                pass
            else:
                raise AssertionError("Invalid reference was accepted")
        print("PASS: CPU diagnostics, locked runtime, isolation, ownership and failure budgets")
    finally:
        subprocess.run(["docker", "image", "rm", IMAGE], check=False, timeout=20)


if __name__ == "__main__":
    main()
