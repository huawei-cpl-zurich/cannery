"""Private model-repository binary storage with immutable revision reads."""

import base64
import json
import math
import os
import subprocess
import sys
from concurrent.futures import Future
from pathlib import Path

from huggingface_hub import HfApi
from pydantic import SecretStr

from cannery.artifacts import ArtifactError, ArtifactRef, digest, verified


class HfArtifactStore:
    def __init__(
        self,
        repository: str,
        token: str,
        api: HfApi | None = None,
        timeout: float = 30,
        endpoint: str | None = None,
    ) -> None:
        if not token.strip():
            raise ArtifactError("Explicit Hugging Face credentials are required")
        # Validate repository identity before any external request.
        ArtifactRef(repository=repository, revision="0" * 40, path="check", sha256="0" * 64)
        self.repository = repository
        if not math.isfinite(timeout) or timeout <= 0:
            raise ArtifactError("Artifact deadline must be positive and finite")
        self.api = api
        self.token = SecretStr(token)
        self.timeout = timeout
        self.endpoint = endpoint

    def _operation(self, payload: dict[str, object]) -> dict[str, object]:
        payload.update(
            repository=self.repository, token=self.token.get_secret_value(), endpoint=self.endpoint
        )
        try:
            result = subprocess.run(
                [sys.executable, "-m", "cannery.artifacts.hf_process"],
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                check=True,
                timeout=self.timeout,
                env={**os.environ, "HF_HUB_DISABLE_XET": "1"},
            )
            response: dict[str, object] = json.loads(result.stdout)
            return response
        except subprocess.TimeoutExpired:
            raise ArtifactError("Artifact operation deadline exceeded") from None
        except (OSError, subprocess.CalledProcessError, ValueError):
            raise ArtifactError("Private artifact operation unavailable") from None

    def _private(self, revision: str | None = None) -> None:
        assert self.api is not None
        info = self.api.repo_info(self.repository, revision=revision, repo_type="model", timeout=5)
        if info.private is not True:
            raise ArtifactError("Artifact repository must be private")
        if revision is not None and info.sha != revision:
            raise ArtifactError("Artifact revision did not resolve exactly")

    def put(self, path: str, data: bytes) -> ArtifactRef:
        reference = ArtifactRef(
            repository=self.repository,
            revision="0" * 40,
            path=path,
            sha256=digest(data),
        )
        if self.api is None:
            result = self._operation(
                {"action": "put", "path": path, "data": base64.b64encode(data).decode()}
            )
            return ArtifactRef.model_validate(result["reference"])
        try:
            self._private()
            commit = self.api.upload_file(
                repo_id=self.repository,
                repo_type="model",
                path_in_repo=path,
                path_or_fileobj=data,
                commit_message="Store CANNery artifact",
            )
            if isinstance(commit, Future):
                raise ArtifactError("Asynchronous artifact upload is unsupported")
            reference = ArtifactRef(**{**reference.model_dump(), "revision": commit.oid})
            self.get(reference)  # Verify pinned bytes before returning a usable pointer.
            return reference
        except ArtifactError:
            raise
        except Exception:
            raise ArtifactError("Private artifact upload unavailable") from None

    def get(self, reference: ArtifactRef) -> bytes:
        if reference.repository != self.repository:
            raise ArtifactError("Artifact belongs to another repository")
        if self.api is None:
            result = self._operation({"action": "get", "reference": reference.model_dump()})
            return verified(base64.b64decode(str(result["data"])), reference)
        try:
            self._private(reference.revision)
            filename = self.api.hf_hub_download(
                repo_id=self.repository,
                repo_type="model",
                filename=reference.path,
                revision=reference.revision,
                etag_timeout=5,
            )
            if not isinstance(filename, str):
                raise ArtifactError("Artifact download did not return a file")
            return verified(Path(filename).read_bytes(), reference)
        except ArtifactError:
            raise
        except Exception:
            raise ArtifactError("Private artifact download unavailable") from None
