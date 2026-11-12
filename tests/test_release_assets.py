import io
import json
import tarfile

import pytest
from release_fixtures import admission as admission
from release_fixtures import arguments as arguments
from release_fixtures import benchmark as benchmark
from release_fixtures import records as records
from release_fixtures import setup_executor as setup_executor

from cannery.artifacts import ArtifactError
from cannery.releases.assets import InvalidReleaseArtifact, verify_assets
from cannery.releases.eligibility import publication_manifest


def test_binary_and_both_timing_blobs_verified_without_execution(admission):
    engine, key, memory, _ = admission
    manifest = publication_manifest(engine, key)
    calls = []
    package = verify_assets(memory, manifest, lambda: calls.append("fence"))
    assert package == memory.get(manifest.package) and len(calls) == 4
    before = memory.puts
    verify_assets(memory, manifest, lambda: None)
    assert memory.puts == before


def test_temporary_blob_outage_does_not_fabricate_a_release(admission, monkeypatch):
    engine, key, memory, _ = admission
    manifest = publication_manifest(engine, key)

    def offline(_):
        raise ArtifactError("offline")

    monkeypatch.setattr(memory, "get", offline)
    with pytest.raises(ArtifactError):
        verify_assets(memory, manifest, lambda: None)


@pytest.mark.parametrize("part", ["package", "timing_artifact", "reference_artifact"])
def test_corrupt_pinned_blob_is_rejected(admission, part):
    engine, key, memory, _ = admission
    manifest = publication_manifest(engine, key)
    memory.data[getattr(manifest, part).sha256] = b"corrupt"
    with pytest.raises(InvalidReleaseArtifact):
        verify_assets(memory, manifest, lambda: None)


@pytest.mark.parametrize(
    "bad", ["key", "missing_case", "sample_count", "samples", "statistics", "unexpected_tensor"]
)
def test_even_repinned_invalid_samples_never_publish(admission, bad):
    engine, key, memory, _ = admission
    manifest = publication_manifest(engine, key)
    value = json.loads(memory.get(manifest.timing_artifact))
    if bad == "key":
        value["key"] = "a" * 64
    if bad == "missing_case":
        value["timing"]["cases"].pop()
    if bad == "sample_count":
        value["timing"]["cases"][0]["samples_us"].pop()
    if bad == "samples":
        value["timing"]["cases"][0]["samples_us"][0] = float("nan")
    if bad == "statistics":
        value["timing"]["cases"][0]["samples_us"] = [99.0] * 100
    if bad == "unexpected_tensor":
        value["outputs"] = {"data": "not permitted"}
    pin = memory.put("timing.json", json.dumps(value).encode())
    changed = manifest.model_copy(update={"timing_artifact": pin})
    with pytest.raises(InvalidReleaseArtifact):
        verify_assets(memory, changed, lambda: None)


@pytest.mark.parametrize("bad", ["source", "abi", "runtime", "extra_file", "duplicate", "symlink"])
def test_even_repinned_unsafe_or_incompatible_package_never_publishes(admission, bad):
    engine, key, memory, _ = admission
    manifest = publication_manifest(engine, key)
    with tarfile.open(fileobj=io.BytesIO(memory.get(manifest.package)), mode="r:") as archive:
        files = {n: archive.extractfile(n).read() for n in archive.getnames()}
    metadata = json.loads(files["metadata.json"])
    if bad == "source":
        metadata["source_sha256"] = "0" * 64
    if bad == "abi":
        metadata["abi"] = "unknown"
    if bad == "runtime":
        metadata["runtime"]["cann"] = "other"
    files["metadata.json"] = json.dumps(metadata).encode()
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w") as archive:
        for name, body in files.items():
            item = tarfile.TarInfo(name)
            item.size = len(body)
            archive.addfile(item, io.BytesIO(body))
        if bad in ["extra_file", "duplicate", "symlink"]:
            item = tarfile.TarInfo("metadata.json" if bad == "duplicate" else "source.py")
            if bad == "symlink":
                item.type = tarfile.SYMTYPE
                item.linkname = "/tmp/escape"
            archive.addfile(item, io.BytesIO(b""))
    pin = memory.put("binary.tar", out.getvalue())
    changed = manifest.model_copy(update={"package": pin})
    with pytest.raises(InvalidReleaseArtifact):
        verify_assets(memory, changed, lambda: None)


def test_ownership_loss_interrupts_before_further_blob_reads(admission):
    engine, key, memory, _ = admission
    manifest = publication_manifest(engine, key)

    def lost():
        raise RuntimeError("ownership lost")

    with pytest.raises(RuntimeError, match="ownership"):
        verify_assets(memory, manifest, lost)
