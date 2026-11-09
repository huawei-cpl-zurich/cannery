"""Trusted entrypoint executed only inside a restricted disposable container."""

import base64
import contextlib
import importlib
import importlib.util
import itertools
import json
import sys
from pathlib import Path
from types import ModuleType


def load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError("Invalid reference module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    payload = json.load(sys.stdin)
    with open("/dev/null", "w") as quiet, contextlib.redirect_stderr(quiet):
        torch = importlib.import_module("torch")
    python = f"{sys.version_info.major}.{sys.version_info.minor}"
    version = str(torch.__version__)
    if python != payload["environment"]["python"] or version != payload["environment"]["torch"]:
        raise ValueError("Installed runtime does not match environment lock")
    root = Path("/work")
    for name, data in payload["files"].items():
        target = root / name
        if not target.resolve().is_relative_to(root):
            raise ValueError("Unsafe source path")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(base64.b64decode(data))
    source = root / payload["path"]
    sys.path[:0] = [str(source), str(root)]
    definition = payload["spec"]
    count = 0
    # Author prints and tracebacks are not a result protocol or public log.
    with (
        open("/dev/null", "w") as quiet,
        contextlib.redirect_stdout(quiet),
        contextlib.redirect_stderr(quiet),
    ):
        reference = load(source / definition["reference"]["file"], "_cannery_reference")
        cases = load(source / definition["cases"]["file"], "_cannery_cases")
        generate = getattr(cases, definition["cases"]["function"])
        forward = getattr(reference, definition["reference"]["function"])
        for case in itertools.islice(generate(), 3):
            if not isinstance(case, dict) or set(case) != set(definition["interface"]["inputs"]):
                raise ValueError("Case inputs do not match the frozen interface")
            with torch.no_grad():
                result = forward(**case)
            names = definition["interface"]["outputs"]
            if isinstance(result, dict):
                if set(result) != set(names):
                    raise ValueError("Reference output names do not match")
                values = list(result.values())
            else:
                values = list(result) if isinstance(result, (tuple, list)) else [result]
            if len(values) != len(names) or not all(
                isinstance(value, torch.Tensor) for value in values
            ):
                raise ValueError("Reference outputs must match the tensor interface")
            count += 1
    if not count:
        raise ValueError("Specification has no smoke cases")
    print(json.dumps({"python": python, "torch": version, "cases": count}))


if __name__ == "__main__":
    main()
