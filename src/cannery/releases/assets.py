"""Verify pinned private blobs without importing or executing native/source code."""

import io
import json
import math
import tarfile
from collections.abc import Callable
from typing import Any

from cannery.artifacts import ArtifactRef, ArtifactStore, digest
from cannery.benchmark.contract import summarize
from cannery.registry.manifest import unique_pairs
from cannery.releases.contract import ReleaseManifest


class InvalidReleaseArtifact(ValueError):
    pass


def pinned(raw: bytes, ref: ArtifactRef, limit: int) -> None:
    if type(raw) is not bytes or not 0 < len(raw) <= limit or digest(raw) != ref.sha256:
        raise InvalidReleaseArtifact("Release artifact digest or size differs")


def verify_package(raw: bytes, manifest: ReleaseManifest) -> None:
    pinned(raw, manifest.package, 64 * 1024 * 1024)
    try:
        files = {}
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
            for member in archive:
                if member.name not in {"metadata.json", "kernel.so"} or member.name in files:
                    raise ValueError("Unexpected package member")
                limit = 16384 if member.name == "metadata.json" else 63 * 1024 * 1024
                if not member.isfile() or not 0 < member.size <= limit:
                    raise ValueError("Invalid member")
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError("Missing member")
                files[member.name] = stream.read(limit + 1)
                if len(files[member.name]) != member.size:
                    raise ValueError("Truncated member")
        if set(files) != {"metadata.json", "kernel.so"}:
            raise ValueError("Incomplete package")
        metadata = json.loads(files["metadata.json"], object_pairs_hook=unique_pairs)
        library = files["kernel.so"]
        static = manifest.language == "catlass"
        expected = {
            "schema_version",
            "device",
            "abi",
            "source_sha256",
            "library_sha256",
            "runtime",
            "compiler",
        }
        if static:
            expected.add("specializations")
        if (
            set(metadata) != expected
            or type(metadata["schema_version"]) is not int
            or metadata["schema_version"] != (2 if static else 1)
            or metadata["device"] != "a5"
            or metadata["abi"] != manifest.abi
            or metadata["runtime"] != manifest.cohort.runtime
            or metadata["source_sha256"] != manifest.source_sha256
            or metadata["library_sha256"] != digest(library)
            or not isinstance(metadata["compiler"], str)
            or not metadata["compiler"]
            or len(library) < 64
            or library[:6] != b"\x7fELF\x02\x01"
            or int.from_bytes(library[16:18], "little") != 3
            or int.from_bytes(library[18:20], "little") != 183
        ):
            raise ValueError("Package contract differs")
        if static and metadata["specializations"] != [
            s.model_dump(mode="json") for s in manifest.specializations or ()
        ]:
            raise ValueError("Specializations differ")
    except (ValueError, KeyError, TypeError, AttributeError, tarfile.TarError, RecursionError):
        raise InvalidReleaseArtifact("Invalid published binary package") from None


def timing_statistics(
    raw: bytes, ref: ArtifactRef, manifest: ReleaseManifest
) -> dict[str, dict[str, Any]]:
    pinned(raw, ref, 256 * 1024)
    try:
        value = json.loads(raw, object_pairs_hook=unique_pairs)
        if set(value) != {"key", "timing"} or value["key"] != manifest.reference_key:
            raise ValueError
        timing = value["timing"]
        cases = timing["cases"]
        expected = [c.id for c in manifest.score.cases if math.prod(c.shape)]
        if (
            set(timing) != {"protocol", "warmup", "cases"}
            or timing["protocol"] != "sync-call-v1"
            or type(timing["warmup"]) is not int
            or timing["warmup"] != 20
            or not isinstance(cases, list)
            or [c["id"] for c in cases] != expected
        ):
            raise ValueError
        result = {}
        for case in cases:
            if set(case) != {"id", "samples_us"}:
                raise ValueError
            result[case["id"]] = summarize(case["samples_us"])
        return result
    except (ValueError, KeyError, TypeError, OverflowError, RecursionError):
        raise InvalidReleaseArtifact("Invalid published timing samples") from None


def verify_assets(
    store: ArtifactStore, manifest: ReleaseManifest, fence: Callable[[], None]
) -> bytes:
    fence()
    package = store.get(manifest.package)
    verify_package(package, manifest)
    fence()
    actual = timing_statistics(
        store.get(manifest.timing_artifact), manifest.timing_artifact, manifest
    )
    fence()
    reference = timing_statistics(
        store.get(manifest.reference_artifact), manifest.reference_artifact, manifest
    )
    for case in manifest.score.cases:
        if case.excluded:
            continue
        assert case.candidate is not None and case.reference is not None
        if actual[case.id] != case.candidate.model_dump(mode="json") or reference[
            case.id
        ] != case.reference.model_dump(mode="json"):
            raise InvalidReleaseArtifact("Timing samples differ from saved statistics")
    fence()
    return package
