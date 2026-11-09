"""Disposable SDK process. Secrets travel over stdin, never process arguments."""

import base64
import json
import sys

import httpx
from huggingface_hub import HfApi, set_client_factory

from cannery.artifacts import ArtifactRef
from cannery.artifacts.hf import HfArtifactStore


def main() -> None:
    payload = json.load(sys.stdin)
    set_client_factory(lambda: httpx.Client(timeout=5, follow_redirects=True))
    store = HfArtifactStore(
        payload["repository"],
        payload["token"],
        api=HfApi(token=payload["token"], endpoint=payload["endpoint"]),
    )
    result: dict[str, object]
    if payload["action"] == "put":
        reference = store.put(payload["path"], base64.b64decode(payload["data"]))
        result = {"reference": reference.model_dump()}
    else:
        data = store.get(ArtifactRef.model_validate(payload["reference"]))
        result = {"data": base64.b64encode(data).decode()}
    print(json.dumps(result))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # SDK diagnostics may contain signed URLs or credentials. Parent returns
        # a redacted error; remote orphan reconciliation is a later milestone.
        sys.exit(1)
