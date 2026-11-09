from cannery.artifacts import ArtifactError, ArtifactStore
from cannery.artifacts.fake import FakeArtifactStore
from cannery.settings import Settings


def make_store(settings: Settings) -> ArtifactStore:
    if settings.artifact_backend == "fake":
        return FakeArtifactStore(settings.artifact_root)
    if not settings.hf_repository or not settings.hf_token:
        raise ArtifactError("HF backend requires a repository and explicit credentials")
    from cannery.artifacts.hf import HfArtifactStore

    return HfArtifactStore(settings.hf_repository, settings.hf_token.get_secret_value())
