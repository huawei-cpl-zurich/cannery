import json

import pytest
import yaml
from spec_fixtures import spec_files

from cannery.registry.manifest import SpecError, parse_manifest, validate_spec


def test_valid_spec_is_content_addressed_without_execution(tmp_path) -> None:
    files = spec_files(reference=b"raise RuntimeError('never execute in intake')")
    first = validate_spec(files, "spec")
    assert first.manifest.kernel.name == "identity"
    assert len(first.content_sha256) == 64
    assert len(first.file_sha256) == 4
    assert validate_spec(dict(reversed(list(files.items()))), "spec") == first
    files["spec/reference.py"] += b"# change"
    assert validate_spec(files, "spec").content_sha256 != first.content_sha256


def test_description_and_version_do_not_change_semantic_digest() -> None:
    files = spec_files()
    before = validate_spec(files, "spec")
    manifest = yaml.safe_load(files["spec/cannery.yaml"])
    manifest["kernel"]["description"] = "Documentation only"
    manifest["spec"]["version"] = 2
    files["spec/cannery.yaml"] = yaml.safe_dump(manifest).encode()
    assert validate_spec(files, "spec").content_sha256 == before.content_sha256


@pytest.mark.parametrize(
    "data",
    [
        b"a: 1\na: 2",
        b"a: &a [*a]",
        b"!!python/object:os.system {}",
        b"[a, b]",
        b"a: [",
        b"a" * 65537,
    ],
)
def test_unsafe_yaml_rejected(data) -> None:
    with pytest.raises(SpecError):
        parse_manifest(data)


@pytest.mark.parametrize(
    "change",
    [
        {"version": True},
        {"unexpected": 1},
        {"reference": {"file": "../outside", "function": "ref"}},
        {"validation": {"metric": "torch_close", "default_rtol": float("nan"), "default_atol": 0}},
    ],
)
def test_invalid_spec_fields(change) -> None:
    files = spec_files()
    manifest = yaml.safe_load(files["spec/cannery.yaml"])
    manifest["spec"].update(change)
    files["spec/cannery.yaml"] = yaml.safe_dump(manifest).encode()
    with pytest.raises(SpecError):
        validate_spec(files, "spec")


def test_required_files_and_lock_are_verified() -> None:
    files = spec_files()
    del files["spec/reference.py"]
    with pytest.raises(SpecError):
        validate_spec(files, "spec")
    files = spec_files()
    files["spec/environment.lock"] = b'{"schema_version":1,"python":"3.12","python":"3.13"}'
    with pytest.raises(SpecError):
        validate_spec(files, "spec")
    files = spec_files()
    files["spec/environment.lock"] = json.dumps(
        {"schema_version": 1, "python": "system", "torch": "latest"}
    ).encode()
    with pytest.raises(SpecError):
        validate_spec(files, "spec")


@pytest.mark.parametrize("path", [".git/config", "spec/.git/hooks/pre-commit"])
def test_nested_git_administration_paths_rejected(path) -> None:
    files = spec_files()
    files[path] = b"untrusted"
    with pytest.raises(SpecError):
        validate_spec(files, "spec")


@pytest.mark.parametrize("version", [True, 1.0, "1"])
def test_schema_versions_are_strict_integers(version) -> None:
    files = spec_files()
    manifest = yaml.safe_load(files["spec/cannery.yaml"])
    manifest["schema_version"] = version
    files["spec/cannery.yaml"] = yaml.safe_dump(manifest).encode()
    with pytest.raises(SpecError):
        validate_spec(files, "spec")
    files = spec_files()
    lock = json.loads(files["spec/environment.lock"])
    lock["schema_version"] = version
    files["spec/environment.lock"] = json.dumps(lock).encode()
    with pytest.raises(SpecError):
        validate_spec(files, "spec")
