"""Copyable client commands for one immutable published release."""

import json
import shlex
from typing import Any

from fastapi import Request

from cannery.settings import Settings


def origin(request: Request, settings: Settings) -> str:
    return settings.auth_origin if settings.auth_enabled else str(request.base_url).rstrip("/")


def snippets(release: dict[str, Any], endpoint: str, hub_enabled: bool) -> dict[str, str | None]:
    identifier = release["id"]
    runtime_file = f"cannery-runtime-{identifier}.json"
    pair = release["tolerances"][0]
    command = [
        "cannery",
        "get",
        release["kernel"],
        "--url",
        endpoint,
        "--release-id",
        identifier,
        "--device",
        release["device"],
        "--no-detect",
        "--runtime-json",
        runtime_file,
        "--rtol",
        str(pair["rtol"]),
        "--atol",
        str(pair["atol"]),
    ]
    if release["specializations"]:
        shape = release["specializations"][0]["shape"]
        command += ["--shape", ",".join(str(value) for value in shape)]
    cli = (
        f"cat > {shlex.quote(runtime_file)} <<'CANNERY_RUNTIME'\n"
        + json.dumps(release["runtime"], indent=2)
        + "\nCANNERY_RUNTIME\n"
        + shlex.join(command)
    )
    python = None
    if hub_enabled and release["hub"]:
        python = (
            "import os\n"
            f'os.environ["HF_ENDPOINT"] = {endpoint!r}\n\n'
            "from kernels import get_kernel\n\n"
            "kernel = get_kernel(\n"
            f"    {release['hub']['repo_id']!r},\n"
            f"    revision={release['hub']['release_revision']!r},\n"
            '    backend="cann",\n    trust_remote_code=True,\n)'
        )
    return {"python": python, "cli": cli}
