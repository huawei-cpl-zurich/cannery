"""Private model-repository binary storage with immutable revision reads."""

from concurrent.futures import Future
from pathlib import Path

import httpx
from huggingface_hub import HfApi, constants, set_client_factory

from cannery.artifacts import ArtifactError, ArtifactRef, digest, verified


class HfArtifactStore:
    def __init__(self, repository: str, token: str, api: HfApi | None = None) -> None:
        if not token.strip():
            raise ArtifactError("Explicit Hugging Face credentials are required")
        # Validate repository identity before any external request.
        ArtifactRef(repository=repository, revision="0" * 40, path="check", sha256="0" * 64)
        self.repository = repository
        if api is None:
            if not constants.HF_HUB_DISABLE_XET:
                raise ArtifactError("Start with HF_HUB_DISABLE_XET=1 for bounded HTTP transfers")
            # The SDK's default shared client has no default request timeout.
            # This process owns HF access; all SDK HTTP phases have finite bounds.
            set_client_factory(lambda: httpx.Client(timeout=5, follow_redirects=True))
        self.api = api if api is not None else HfApi(token=token)

    def _private(self, revision: str | None = None) -> None:
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
