import json

import pytest
from test_build_executor import Memory
from test_release_contract import release_data as release_data
from test_tensor_release_contract import parse
from test_tensor_release_contract import tensor_release as tensor_release

from cannery.artifacts import ArtifactError, digest
from cannery.builds.library import archive_library
from cannery.releases.assets import InvalidReleaseArtifact, verify_assets
from cannery.releases.tensor_contract import read_tensor_manifest


@pytest.fixture
def assets(tensor_release):
    memory = Memory()
    manifest = parse(tensor_release)
    library = bytearray(64)
    library[:6] = b"\x7fELF\x02\x01"
    library[16:20] = b"\x03\0\xb7\0"
    metadata = dict(
        schema_version=3,
        abi="tensor-v2",
        device="a5",
        runtime=manifest.cohort.runtime,
        compiler="pinned offline compiler",
        source_sha256=manifest.source_sha256,
        library_sha256=digest(bytes(library)),
        variants=manifest.variants(),
    )
    package = archive_library(bytes(library), metadata, metadata_limit=256 * 1024)
    pin = memory.put("package.tar", package)
    timing = json.dumps(
        {
            "key": manifest.reference_key,
            "timing": {
                "protocol": "sync-call-tensor-v2",
                "warmup": 20,
                "cases": [{"id": "small", "samples_us": [10.0] * 94 + [12.0] * 6}],
            },
        }
    ).encode()
    samples = memory.put("timing.json", timing)
    value = manifest.model_dump(mode="json") | {
        "package": pin.model_dump(mode="json"),
        "timing_artifact": samples.model_dump(mode="json"),
        "reference_artifact": samples.model_dump(mode="json"),
    }
    return memory, parse(value), metadata, bytes(library)


def test_tensor_package_and_samples_verified_without_execution_after_canonical_roundtrip(assets):
    memory, manifest, _, _ = assets
    raw = manifest.encoded()
    restored = read_tensor_manifest(raw, digest(raw))
    assert list(restored.cases) != list(manifest.cases)
    assert restored.variants() == manifest.variants()
    calls = []
    assert verify_assets(memory, restored, lambda: calls.append(1)) == memory.get(manifest.package)
    assert len(calls) == 4


@pytest.mark.parametrize("fault", ["variants", "scalar", "alias", "runtime", "schema", "compiler"])
def test_repinned_incompatible_tensor_package_rejected(assets, fault):
    memory, manifest, metadata, library = assets
    if fault == "variants":
        metadata["variants"].reverse()
    elif fault == "scalar":
        metadata["variants"][0]["scalars"][0]["value"] = 99
    elif fault == "alias":
        metadata["variants"][0]["outputs"][0]["alias"] = None
    elif fault == "runtime":
        metadata["runtime"] = metadata["runtime"] | {"cann": "other"}
    elif fault == "schema":
        metadata["schema_version"] = 2
    else:
        metadata["compiler"] = "x" * 8193
    pin = memory.put("wrong.tar", archive_library(library, metadata, metadata_limit=256 * 1024))
    with pytest.raises(InvalidReleaseArtifact):
        verify_assets(memory, manifest.model_copy(update={"package": pin}), lambda: None)


@pytest.mark.parametrize("fault", ["protocol", "empty_case", "statistics", "outputs"])
def test_invalid_or_output_bearing_tensor_timings_rejected(assets, fault):
    memory, manifest, _, _ = assets
    value = json.loads(memory.get(manifest.timing_artifact))
    if fault == "protocol":
        value["timing"]["protocol"] = "sync-call-v1"
    elif fault == "empty_case":
        value["timing"]["cases"].append({"id": "empty", "samples_us": [10.0] * 100})
    elif fault == "statistics":
        value["timing"]["cases"][0]["samples_us"] = [99.0] * 100
    else:
        value["outputs"] = {"small": [1, 2]}
    pin = memory.put("wrong.json", json.dumps(value).encode())
    with pytest.raises(InvalidReleaseArtifact):
        verify_assets(memory, manifest.model_copy(update={"timing_artifact": pin}), lambda: None)


def test_storage_outage_and_fencing_do_not_publish_or_execute(assets):
    memory, manifest, _, _ = assets
    memory.get = lambda _: (_ for _ in ()).throw(ArtifactError("offline"))
    with pytest.raises(ArtifactError):
        verify_assets(memory, manifest, lambda: None)

    def lost():
        raise RuntimeError("lease lost")

    with pytest.raises(RuntimeError, match="lease lost"):
        verify_assets(memory, manifest, lost)
