import pytest
from pydantic import ValidationError

from cannery.artifacts import ArtifactError, ArtifactRef
from cannery.artifacts.fake import FakeArtifactStore


def test_immutable_roundtrip_and_corruption(tmp_path) -> None:
    store = FakeArtifactStore(tmp_path)
    first = store.put("binary/op.so", b"one")
    second = store.put("binary/op.so", b"two")
    assert first.revision != second.revision
    assert store.put("binary/op.so", b"one") == first
    assert store.get(first) == b"one"
    assert store.get(second) == b"two"
    (tmp_path / first.revision / first.path).write_bytes(b"corrupted")
    with pytest.raises(ArtifactError, match="checksum"):
        store.get(first)
    with pytest.raises(ArtifactError, match="checksum"):
        store.put("binary/op.so", b"one")


@pytest.mark.parametrize("path", ["../op", "/op", "x/../op", "x//op", "./op", "", "x\\op"])
def test_unsafe_paths_rejected(tmp_path, path) -> None:
    with pytest.raises(ValidationError):
        FakeArtifactStore(tmp_path).put(path, b"data")


def test_missing_and_foreign_artifacts(tmp_path) -> None:
    store = FakeArtifactStore(tmp_path)
    reference = ArtifactRef(repository="fake", revision="a" * 64, path="op", sha256="a" * 64)
    with pytest.raises(ArtifactError, match="read unavailable"):
        store.get(reference)
    with pytest.raises(ArtifactError, match="another repository"):
        store.get(reference.model_copy(update={"repository": "org/private"}))


def test_symlink_escape_rejected(tmp_path) -> None:
    store = FakeArtifactStore(tmp_path / "store")
    outside = tmp_path / "outside"
    outside.mkdir()
    (store.root / ("a" * 64)).symlink_to(outside, target_is_directory=True)
    reference = ArtifactRef(repository="fake", revision="a" * 64, path="op", sha256="a" * 64)
    with pytest.raises(ArtifactError, match="escapes"):
        store.get(reference)
