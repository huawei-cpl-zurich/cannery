import copy

import pytest
import yaml
from submission_fixtures import submission_files

from cannery.submissions.manifest import SubmissionError, validate_submission


def native_manifest():
    files = submission_files()
    manifest = yaml.safe_load(files["candidate/submission.yaml"])
    manifest["implementation"]["build"].update(
        abi="catlass-unary-f32-v1",
        specializations=[
            {"shape": [17], "dtype": "float32"},
            {"shape": [3, 7], "dtype": "float32"},
            {"shape": [0], "dtype": "float32"},
        ],
    )
    return files, manifest


def validate(files, manifest):
    files = {**files, "candidate/submission.yaml": yaml.safe_dump(manifest).encode()}
    return validate_submission(files, "candidate")


def test_explicit_specializations_are_hashed_without_importing_author_code():
    files, manifest = native_manifest()
    original = validate(files, manifest)
    build = original.manifest.implementation.build
    assert [v.shape for v in build.specializations] == [(17,), (3, 7), (0,)]
    assert build.abi == "catlass-unary-f32-v1"
    manifest["implementation"]["build"]["specializations"][0]["shape"] = [18]
    assert validate(files, manifest).content_sha256 != original.content_sha256
    assert original.manifest.model_dump(mode="json")["implementation"]["build"]["specializations"][
        1
    ] == {"shape": [3, 7], "dtype": "float32"}


@pytest.mark.parametrize(
    "variants",
    [
        [],
        [{"shape": [1], "dtype": "float32"}] * 5,
        [{"shape": [1], "dtype": "float32"}] * 2,
        [{"shape": [], "dtype": "float32"}],
        [{"shape": [1] * 5, "dtype": "float32"}],
        [{"shape": [True], "dtype": "float32"}],
        [{"shape": [1.5], "dtype": "float32"}],
        [{"shape": [-1], "dtype": "float32"}],
        [{"shape": [1048577], "dtype": "float32"}],
        [{"shape": [1024, 1025], "dtype": "float32"}],
        [{"shape": [1], "dtype": "float16"}],
        [{"shape": [1], "dtype": "float32", "command": "arbitrary"}],
        None,
    ],
)
def test_unsupported_or_unbounded_specializations_rejected(variants):
    files, manifest = native_manifest()
    manifest["implementation"]["build"]["specializations"] = variants
    with pytest.raises(SubmissionError):
        validate(files, manifest)


@pytest.mark.parametrize("change", ["a3", "ascendc", "legacy", "debug"])
def test_specialization_contract_cannot_be_attached_to_another_target(change):
    files, manifest = native_manifest()
    impl = manifest["implementation"]
    if change == "a3":
        impl["device"] = "a3"
    elif change == "ascendc":
        impl.update(language="ascendc", entrypoint="kernel.cpp:forward")
        impl["build"].update(adapter="ascendc", inputs=["kernel.cpp"])
        files["candidate/kernel.cpp"] = b"not executed"
    elif change == "legacy":
        del impl["build"]["abi"]
    else:
        impl["build"]["options"]["optimization"] = "debug"
    with pytest.raises(SubmissionError):
        validate(files, manifest)


def test_legacy_catlass_hash_and_serialization_stay_identical():
    files = submission_files()
    manifest = yaml.safe_load(files["candidate/submission.yaml"])
    frozen = validate(files, manifest)
    changed = copy.deepcopy(manifest)
    changed["implementation"]["build"].update(abi=None, specializations=None)
    assert validate(files, changed).content_sha256 == frozen.content_sha256
    assert frozen.manifest.model_dump(mode="json") == manifest
