import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from cannery.artifacts import ArtifactError, ArtifactRef, digest, verified


class FakeArtifactStore:
    """Private local test store with content-addressed immutable revisions."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _target(self, reference: ArtifactRef) -> Path:
        if reference.repository != "fake":
            raise ArtifactError("Artifact belongs to another repository")
        path = self.root / reference.revision / reference.path
        if not path.resolve().is_relative_to(self.root):
            raise ArtifactError("Artifact path escapes storage root")
        return path

    def put(self, path: str, data: bytes) -> ArtifactRef:
        checksum = digest(data)
        reference = ArtifactRef(repository="fake", revision=checksum, path=path, sha256=checksum)
        target = self._target(reference)
        temporary = None
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            # Publish a complete file atomically without replacing an existing
            # immutable revision. Concurrent readers never observe partial data.
            with NamedTemporaryFile(dir=target.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(data)
            try:
                os.link(temporary, target)
            except FileExistsError:
                verified(target.read_bytes(), reference)
        except OSError:
            raise ArtifactError("Artifact write unavailable") from None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return reference

    def get(self, reference: ArtifactRef) -> bytes:
        try:
            return verified(self._target(reference).read_bytes(), reference)
        except OSError:
            raise ArtifactError("Artifact read unavailable") from None
