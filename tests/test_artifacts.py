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


def test_failed_write_cleans_partial_temporary_file(tmp_path, monkeypatch) -> None:
    from tempfile import NamedTemporaryFile

    def partial_file(*args, **kwargs):
        stream = NamedTemporaryFile(*args, **kwargs)
        original = stream.write

        def fail(data):
            original(data[:2])
            raise OSError("disk full")

        stream.write = fail
        return stream

    monkeypatch.setattr("cannery.artifacts.fake.NamedTemporaryFile", partial_file)
    with pytest.raises(ArtifactError, match="write unavailable"):
        FakeArtifactStore(tmp_path).put("op", b"binary")
    assert not [path for path in tmp_path.rglob("*") if path.is_file()]


def test_concurrent_puts_publish_complete_immutable_file(tmp_path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    store = FakeArtifactStore(tmp_path)
    data = b"binary" * 10000
    with ThreadPoolExecutor(max_workers=4) as pool:
        refs = list(pool.map(lambda _: store.put("op", data), range(8)))
    assert all(reference == refs[0] for reference in refs)
    assert store.get(refs[0]) == data
