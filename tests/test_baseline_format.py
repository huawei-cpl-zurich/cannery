import hashlib
import io
import json
import tarfile

import pytest
from test_recipe_spec import recipe_spec

from cannery.numerics.baseline_format import baseline_key, pack_baseline, unpack_baseline
from cannery.numerics.recipes import generate, parse_recipes
from cannery.registry.manifest import validate_spec


def fixture():
    files, _, _ = recipe_spec()
    spec = validate_spec(files, "spec")
    suite = parse_recipes(files["spec/recipes.yaml"])
    outputs = {c.id: generate(next(iter(c.inputs.values())), c.id, "x") for c in suite.cases}
    return spec, suite, outputs


def bundle(spec, suite, outputs):
    return pack_baseline(
        spec.content_sha256, suite, spec.environment, "sha256:" + "a" * 64, outputs
    )


def unpack(package, spec, recipe_suite, **changes):
    return unpack_baseline(
        package,
        hashlib.sha256(package).hexdigest(),
        **(
            {
                "spec_sha256": spec.content_sha256,
                "suite": recipe_suite,
                "environment": spec.environment,
            }
            | changes
        ),
    )


def rewrite(package, extra=None, mutate=None):
    stream = io.BytesIO()
    with (
        tarfile.open(fileobj=io.BytesIO(package)) as source,
        tarfile.open(fileobj=stream, mode="w") as target,
    ):
        for member in source:
            data = source.extractfile(member).read()
            if mutate:
                data = mutate(member.name, data)
            member.size = len(data)
            target.addfile(member, io.BytesIO(data))
        if extra:
            name, kind = extra
            item = tarfile.TarInfo(name)
            item.type = kind
            item.linkname = "/etc/passwd"
            target.addfile(item, io.BytesIO(b""))
    return stream.getvalue()


def test_deterministic_private_package_has_outputs_and_provenance_but_no_inputs():
    spec, suite, outputs = fixture()
    package = bundle(spec, suite, outputs)
    assert package == bundle(spec, suite, dict(reversed(list(outputs.items()))))
    decoded = unpack(package, spec, suite)
    assert decoded.outputs == outputs
    assert decoded.metadata["key"] == baseline_key(spec.content_sha256, suite, spec.environment)
    with tarfile.open(fileobj=io.BytesIO(package)) as archive:
        assert set(archive.getnames()) == {
            "metadata.json",
            "outputs/small.f32",
            "outputs/strided.f32",
            "outputs/empty.f32",
        }
        assert archive.getmember("outputs/empty.f32").size == 0
    assert all("input_sha256" in c for c in decoded.metadata["cases"])
    assert "input_bytes" not in json.dumps(decoded.metadata)


@pytest.mark.parametrize("change", ["spec", "environment", "suite"])
def test_cache_cannot_be_rebound_to_other_spec_environment_or_recipes(change):
    spec, suite, outputs = fixture()
    package = bundle(spec, suite, outputs)
    changes = {}
    if change == "spec":
        changes["spec_sha256"] = "0" * 64
    elif change == "environment":
        changes["environment"] = spec.environment.model_copy(update={"torch": "2.7.0+cpu"})
    else:
        data = suite.model_dump(mode="json")
        data["cases"][0]["inputs"]["x"]["seed"] = 43
        changes["suite"] = parse_recipes(json.dumps(data).encode())
    with pytest.raises(ValueError):
        unpack(package, spec, suite, **changes)


@pytest.mark.parametrize(
    "extra",
    [
        ("metadata.json", tarfile.REGTYPE),
        ("../escape", tarfile.REGTYPE),
        ("inputs/x.f32", tarfile.REGTYPE),
        ("outputs/link", tarfile.SYMTYPE),
    ],
)
def test_unexpected_duplicate_or_link_members_are_rejected(extra):
    spec, suite, outputs = fixture()
    package = rewrite(bundle(spec, suite, outputs), extra=extra)
    with pytest.raises(ValueError):
        unpack(package, spec, suite)


@pytest.mark.parametrize("kind", ["metadata", "bytes", "shape", "missing"])
def test_corrupt_outputs_descriptors_and_provenance_fail_closed(kind):
    spec, suite, outputs = fixture()
    package = bundle(spec, suite, outputs)

    def mutate(name, data):
        if name == "metadata.json" and kind in ("metadata", "shape"):
            meta = json.loads(data)
            if kind == "metadata":
                meta["generator"] = "unknown"
            else:
                meta["cases"][0]["shape"] = [6]
            return json.dumps(meta).encode()
        if name == "outputs/small.f32" and kind == "bytes":
            return bytes([data[0] ^ 1]) + data[1:]
        if name == "outputs/small.f32" and kind == "missing":
            return b""
        return data

    with pytest.raises(ValueError):
        unpack(rewrite(package, mutate=mutate), spec, suite)


def test_wrong_digest_and_incomplete_output_suite_never_publish():
    spec, suite, outputs = fixture()
    package = bundle(spec, suite, outputs)
    with pytest.raises(ValueError):
        unpack_baseline(
            package,
            "0" * 64,
            spec_sha256=spec.content_sha256,
            suite=suite,
            environment=spec.environment,
        )
    outputs.pop("empty")
    with pytest.raises(ValueError):
        bundle(spec, suite, outputs)
    with pytest.raises(ValueError):
        pack_baseline(spec.content_sha256, suite, spec.environment, "latest", outputs)
