"""Dispatch bounded immutable release documents while preserving schema-1 bytes."""

import json
from typing import Any

from cannery.artifacts import digest
from cannery.registry.manifest import unique_pairs
from cannery.releases.contract import ReleaseManifest, read_manifest
from cannery.releases.tensor_contract import TensorReleaseManifest, read_tensor_manifest

ReleaseDocument = ReleaseManifest | TensorReleaseManifest


def parse_release_document(value: dict[str, Any]) -> ReleaseDocument:
    model = TensorReleaseManifest if value.get("schema_version") == 2 else ReleaseManifest
    return model.model_validate_json(json.dumps(value))


def read_release_document(raw: bytes, sha256: str) -> ReleaseDocument:
    if type(raw) is not bytes or not 0 < len(raw) <= 256 * 1024 or digest(raw) != sha256:
        raise ValueError("Release document digest or size mismatch")
    try:
        value = json.loads(raw, object_pairs_hook=unique_pairs)
        if not isinstance(value, dict):
            raise ValueError("Expected a release object")
        reader = read_tensor_manifest if value.get("schema_version") == 2 else read_manifest
        return reader(raw, sha256)
    except (ValueError, TypeError, RecursionError, OverflowError) as error:
        raise ValueError("Invalid release document") from error
