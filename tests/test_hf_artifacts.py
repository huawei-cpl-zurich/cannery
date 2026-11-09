from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

from cannery.artifacts import ArtifactError
from cannery.artifacts.factory import make_store
from cannery.artifacts.hf import HfArtifactStore
from cannery.settings import Settings


def fixture_store(tmp_path):
    api = MagicMock()
    api.repo_info.return_value = SimpleNamespace(private=True, sha="a" * 40)
    api.upload_file.return_value = SimpleNamespace(oid="a" * 40)
    path = tmp_path / "download"
    path.write_bytes(b"binary")
    api.hf_hub_download.return_value = str(path)
    return HfArtifactStore("org/private", "explicit-token", api=api), api, path


def test_private_pinned_verified_roundtrip(tmp_path) -> None:
    store, api, _ = fixture_store(tmp_path)
    reference = store.put("op.so", b"binary")
    assert store.get(reference) == b"binary"
    assert reference.revision == "a" * 40
    assert api.hf_hub_download.call_args.kwargs["revision"] == reference.revision
    assert api.upload_file.call_args.kwargs["path_or_fileobj"] == b"binary"
    assert api.repo_info.call_args.kwargs["timeout"] == 5


def test_public_repository_rejected_before_upload(tmp_path) -> None:
    store, api, _ = fixture_store(tmp_path)
    api.repo_info.return_value.private = False
    with pytest.raises(ArtifactError, match="private"):
        store.put("op.so", b"binary")
    api.upload_file.assert_not_called()


@pytest.mark.parametrize(
    "failure", [TimeoutError("secret"), httpx.ReadError("secret"), PermissionError("secret")]
)
def test_upload_errors_are_finite_and_redacted(tmp_path, failure) -> None:
    store, api, _ = fixture_store(tmp_path)
    api.upload_file.side_effect = failure
    with pytest.raises(ArtifactError, match="upload unavailable") as caught:
        store.put("op.so", b"binary")
    assert "secret" not in str(caught.value)
    assert api.upload_file.call_count == 1


def test_interrupted_and_corrupt_downloads(tmp_path) -> None:
    store, api, path = fixture_store(tmp_path)
    reference = store.put("op.so", b"binary")
    api.hf_hub_download.side_effect = httpx.ReadTimeout("secret")
    with pytest.raises(ArtifactError, match="download unavailable"):
        store.get(reference)
    api.hf_hub_download.side_effect = None
    path.write_bytes(b"partial")
    with pytest.raises(ArtifactError, match="checksum"):
        store.get(reference)
    api.repo_info.return_value.sha = "b" * 40
    with pytest.raises(ArtifactError, match="resolve exactly"):
        store.get(reference)


def test_fake_default_and_explicit_hf_configuration(tmp_path) -> None:
    assert make_store(Settings(artifact_root=tmp_path)).put("op", b"data")
    with pytest.raises(ArtifactError, match="explicit credentials"):
        make_store(Settings(artifact_backend="hf", hf_repository="org/private"))
    with pytest.raises(ArtifactError, match="credentials"):
        HfArtifactStore("org/private", " ")


def test_operation_timeout_kills_subprocess(monkeypatch) -> None:
    import subprocess

    def timeout(*args, **kwargs):
        assert "token" not in " ".join(args[0])
        assert kwargs["timeout"] == 0.1
        raise subprocess.TimeoutExpired(args[0], 0.1)

    monkeypatch.setattr("cannery.artifacts.hf.subprocess.run", timeout)
    store = HfArtifactStore("org/private", "token", timeout=0.1)
    with pytest.raises(ArtifactError, match="deadline exceeded"):
        store.put("op", b"data")
