from cannery.artifacts import ArtifactError, ArtifactStore
from cannery.artifacts.fake import FakeArtifactStore
from cannery.settings import Settings


def make_store(settings: Settings) -> ArtifactStore:
    if settings.artifact_backend == "fake":
        return FakeArtifactStore(settings.artifact_root)
    if settings.artifact_backend == "private":
        if not settings.artifact_read_token:
            raise ArtifactError("Private backend requires explicit reader credentials")
        from cannery.artifacts.private import PrivateArtifactStore

        return PrivateArtifactStore(
            settings.artifact_url, settings.artifact_read_token.get_secret_value(),
            write_token=(settings.artifact_write_token.get_secret_value()
                         if settings.artifact_write_token else None),
        )
    if not settings.hf_repository or not settings.hf_token:
        raise ArtifactError("HF backend requires a repository and explicit credentials")
    from cannery.artifacts.hf import HfArtifactStore

    return HfArtifactStore(settings.hf_repository, settings.hf_token.get_secret_value())
