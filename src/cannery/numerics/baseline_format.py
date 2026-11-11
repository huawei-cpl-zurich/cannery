"""Private CPU oracle outputs: bounded raw float32 tensors, never pickle."""

import hashlib
import io
import json
import math
import tarfile
from dataclasses import dataclass
from typing import Any

from cannery.numerics.recipes import MAX_SUITE_BYTES, RecipeSuite, generate, input_sha256
from cannery.registry.manifest import EnvironmentLock

MAX_PACKAGE_BYTES = MAX_SUITE_BYTES + 128 * 1024


def baseline_key(spec_sha256: str, suite: RecipeSuite, environment: EnvironmentLock) -> str:
    if len(spec_sha256) != 64 or any(c not in "0123456789abcdef" for c in spec_sha256):
        raise ValueError("Invalid frozen spec identity")
    value = {
        "spec_sha256": spec_sha256,
        "suite_sha256": suite.sha256,
        "generator": suite.generator,
        "environment": environment.model_dump(mode="json"),
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def records(suite: RecipeSuite, outputs: dict[str, bytes]) -> list[dict[str, Any]]:
    if set(outputs) != {case.id for case in suite.cases}:
        raise ValueError("Baseline must cover every mandatory case exactly once")
    result, total = [], 0
    for case in suite.cases:
        name, recipe = next(iter(case.inputs.items()))
        output = outputs[case.id]
        if type(output) is not bytes or len(output) != 4 * math.prod(recipe.shape):
            raise ValueError("Reference output must preserve float32 input shape")
        total += len(output)
        if total > MAX_SUITE_BYTES:
            raise ValueError("Baseline exceeds aggregate output budget")
        result.append(
            {
                "id": case.id,
                "shape": list(recipe.shape),
                "dtype": "float32",
                "input_sha256": input_sha256(recipe, generate(recipe, case.id, name)),
                "output_sha256": hashlib.sha256(output).hexdigest(),
            }
        )
    return result


@dataclass(frozen=True)
class BaselineBundle:
    metadata: dict[str, Any]
    outputs: dict[str, bytes]


def pack_baseline(
    spec_sha256: str,
    suite: RecipeSuite,
    environment: EnvironmentLock,
    image_id: str,
    outputs: dict[str, bytes],
) -> bytes:
    if (
        not image_id.startswith("sha256:")
        or len(image_id) != 71
        or any(c not in "0123456789abcdef" for c in image_id[7:])
    ):
        raise ValueError("Expected immutable oracle image ID")
    metadata = {
        "schema_version": 1,
        "key": baseline_key(spec_sha256, suite, environment),
        "spec_sha256": spec_sha256,
        "suite_sha256": suite.sha256,
        "generator": suite.generator,
        "environment": environment.model_dump(mode="json"),
        "image_id": image_id,
        "cases": records(suite, outputs),
    }
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    if len(encoded) > 65536:
        raise ValueError("Baseline metadata exceeds budget")
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name, value in [
            ("metadata.json", encoded),
            *[(f"outputs/{c.id}.f32", outputs[c.id]) for c in suite.cases],
        ]:
            item = tarfile.TarInfo(name)
            item.size, item.mode = len(value), 0o400
            archive.addfile(item, io.BytesIO(value))
    package = stream.getvalue()
    if len(package) > MAX_PACKAGE_BYTES:
        raise ValueError("Baseline package exceeds budget")
    return package


def unpack_baseline(
    package: bytes,
    sha256: str,
    *,
    spec_sha256: str,
    suite: RecipeSuite,
    environment: EnvironmentLock,
) -> BaselineBundle:
    if (
        type(package) is not bytes
        or len(package) > MAX_PACKAGE_BYTES
        or hashlib.sha256(package).hexdigest() != sha256
    ):
        raise ValueError("Baseline digest or size mismatch")
    try:
        names = {"metadata.json"} | {f"outputs/{c.id}.f32" for c in suite.cases}
        contents = {}
        with tarfile.open(fileobj=io.BytesIO(package), mode="r:") as archive:
            for item in archive:
                if item.name not in names or item.name in contents or not item.isfile():
                    raise ValueError("Unexpected or duplicate baseline member")
                limit = 65536 if item.name == "metadata.json" else 4 * 1048576
                if not 0 <= item.size <= limit:
                    raise ValueError("Baseline member exceeds budget")
                source = archive.extractfile(item)
                if source is None:
                    raise ValueError("Missing baseline member")
                contents[item.name] = source.read(limit + 1)
                if len(contents[item.name]) != item.size:
                    raise ValueError("Truncated baseline member")
        if set(contents) != names:
            raise ValueError("Missing baseline cases")
        metadata = json.loads(contents["metadata.json"])
        outputs = {c.id: contents[f"outputs/{c.id}.f32"] for c in suite.cases}
        # Regenerate the exact trusted metadata, including input and output hashes.
        canonical = pack_baseline(spec_sha256, suite, environment, metadata["image_id"], outputs)
        with tarfile.open(fileobj=io.BytesIO(canonical), mode="r:") as archive:
            member = archive.extractfile("metadata.json")
            assert member is not None
            expected = member.read()
        if contents["metadata.json"] != expected:
            raise ValueError("Baseline provenance or descriptors mismatch")
        return BaselineBundle(metadata, outputs)
    except (tarfile.TarError, KeyError, TypeError, AttributeError, OverflowError, RecursionError):
        raise ValueError("Invalid cached baseline package") from None
