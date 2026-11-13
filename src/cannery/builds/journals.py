"""Keep shared cleanup journals owned by the configured worker's device route."""

import json
from pathlib import Path

from cannery.builds.sandbox import WorkerProfile


def retained_profile(journal: Path, owner: WorkerProfile | None) -> WorkerProfile | None:
    context = journal / "context.json"
    if not context.exists():
        return None
    saved = WorkerProfile.model_validate_json(
        json.dumps(json.loads(context.read_bytes())["profile"])
    )
    if owner is not None and any(
        getattr(saved, name) != getattr(owner, name)
        for name in ("device", "target", "workspace", "npu")
    ):
        return None
    return saved
