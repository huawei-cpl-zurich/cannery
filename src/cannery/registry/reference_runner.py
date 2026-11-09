"""Trusted supervisor; author modules execute only in a separate non-root process."""

import base64
import importlib.metadata
import json
import os
import secrets
import signal
import subprocess
import sys
from pathlib import Path

# This helper's replies are untrusted data, not completion evidence. Only the
# supervisor counts successful exchanges and owns the container's stdout.
WORKER = r"""
import contextlib, importlib.util, json, sys
from pathlib import Path
import torch


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def reply(request, **fields):
    print(json.dumps(dict(nonce=request['nonce'], **fields)), flush=True)


request = json.loads(sys.stdin.readline())
root = Path('/work') / request['path']
definition = request['spec']
sys.path[:0] = [str(root), '/work']
with open('/dev/null', 'w') as quiet, contextlib.redirect_stdout(quiet):
    reference = load(root / definition['reference']['file'], '_reference')
    cases = load(root / definition['cases']['file'], '_cases')
    forward = getattr(reference, definition['reference']['function'])
    generate = getattr(cases, definition['cases']['function'])
    if not callable(forward) or not callable(generate):
        raise ValueError('Missing callable')
    iterator = iter(generate())
reply(request, ready=True)
for line in sys.stdin:
    request = json.loads(line)
    with open('/dev/null', 'w') as quiet, contextlib.redirect_stdout(quiet):
        if request['op'] == 'next':
            try:
                case = next(iterator)
            except StopIteration:
                result = dict(done=True)
            else:
                if not isinstance(case, dict):
                    raise ValueError('Cases must yield dictionaries')
                result = dict(done=False, inputs=list(case))
        elif request['op'] == 'call':
            with torch.no_grad():
                output = forward(**case)
            if isinstance(output, dict):
                if set(output) != set(definition['interface']['outputs']):
                    raise ValueError('Output names do not match')
                values = list(output.values())
            else:
                values = list(output) if isinstance(output, (tuple, list)) else [output]
            if not all(isinstance(value, torch.Tensor) for value in values):
                raise ValueError('Outputs must be tensors')
            result = dict(outputs=[dict(shape=list(value.shape), dtype=str(value.dtype))
                                   for value in values])
        else:
            raise ValueError('Unknown operation')
    reply(request, **result)
"""


def demote() -> None:
    os.setgroups([])
    os.setgid(10001)
    os.setuid(10001)


def main() -> None:
    signal.signal(signal.SIGALRM, lambda *_: os._exit(124))
    signal.alarm(70)
    payload = json.load(sys.stdin)
    signal.setitimer(signal.ITIMER_REAL, payload["timeout"] + 2)
    python = f"{sys.version_info.major}.{sys.version_info.minor}"
    torch = importlib.metadata.version("torch")
    if python != payload["environment"]["python"] or torch != payload["environment"]["torch"]:
        raise ValueError("Installed runtime does not match environment lock")
    root = Path("/work")
    for name, data in payload["files"].items():
        target = root / name
        if not target.resolve().is_relative_to(root):
            raise ValueError("Unsafe source path")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(base64.b64decode(data))
    # A distinct UID prevents author code from opening /proc/<supervisor>/fd/1
    # or its memory. The source tree is root-owned and not writable by the child.
    worker = subprocess.Popen(
        [sys.executable, "-I", "-c", WORKER],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        preexec_fn=demote,
        env={
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": "/tmp",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        },
    )
    count = 0
    try:

        def exchange(operation: str, **fields: object) -> dict[str, object]:
            nonce = secrets.token_hex(16)
            assert worker.stdin is not None and worker.stdout is not None
            worker.stdin.write(json.dumps(dict(nonce=nonce, op=operation, **fields)) + "\n")
            worker.stdin.flush()
            result = json.loads(worker.stdout.readline(65537))
            if not isinstance(result, dict) or result.get("nonce") != nonce:
                raise ValueError("Invalid worker reply")
            return result

        definition = payload["spec"]
        if exchange("init", path=payload["path"], spec=definition).get("ready") is not True:
            raise ValueError("Worker failed to initialize")
        for _ in range(3):
            case = exchange("next")
            if case.get("done") is True:
                break
            inputs = case.get("inputs")
            if (
                case.get("done") is not False
                or not isinstance(inputs, list)
                or not all(isinstance(name, str) for name in inputs)
                or len(inputs) != len(definition["interface"]["inputs"])
                or set(inputs) != set(definition["interface"]["inputs"])
            ):
                raise ValueError("Case inputs do not match interface")
            outputs = exchange("call").get("outputs")
            if not isinstance(outputs, list) or len(outputs) != len(
                definition["interface"]["outputs"]
            ):
                raise ValueError("Output interface mismatch")
            for output in outputs:
                if (
                    not isinstance(output, dict)
                    or not isinstance(output.get("shape"), list)
                    or not isinstance(output.get("dtype"), str)
                    or not output["dtype"].startswith("torch.")
                ):
                    raise ValueError("Invalid tensor metadata")
            count += 1
        if not count:
            raise ValueError("Specification has no smoke cases")
    finally:
        worker.kill()
        worker.wait(timeout=2)
    print(json.dumps({"python": python, "torch": torch, "cases": count}))


if __name__ == "__main__":
    main()
