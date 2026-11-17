import copy

import pytest
import yaml
from submission_fixtures import submission_files

from cannery.submissions.manifest import (
    SubmissionError,
    SubmissionManifest,
    validate_submission,
    validate_tensor_submission,
)
from cannery.submissions.models import RequestedTarget


def tensor_files(language="triton"):
    files = submission_files("catlass" if language != "ascendc" else "ascendc")
    manifest = yaml.safe_load(files["candidate/submission.yaml"])
    manifest["schema_version"] = 2
    manifest["implementation"]["language"] = language
    manifest["implementation"]["build"].update(adapter=language, abi="tensor-v2")
    files["candidate/submission.yaml"] = yaml.safe_dump(manifest).encode()
    return files


@pytest.mark.parametrize("language", ["ascendc", "catlass", "triton"])
def test_all_native_tensor_backends_read_without_executing_source(language):
    files = tensor_files(language)
    result = validate_tensor_submission(files, "candidate")
    assert result.manifest.schema_version == 2
    assert result.manifest.implementation.language == language
    assert result.manifest.implementation.build.abi == "tensor-v2"
    assert validate_tensor_submission(dict(reversed(list(files.items()))), "candidate") == result
    with pytest.raises(SubmissionError):
        validate_submission(files, "candidate")
    files["candidate/submission.yaml"] += b"\n# comment\n"
    assert validate_tensor_submission(files, "candidate").content_sha256 == result.content_sha256
    path = result.manifest.implementation.build.inputs[0]
    files["candidate/" + path] += b"\n# source change\n"
    assert validate_tensor_submission(files, "candidate").content_sha256 != result.content_sha256


@pytest.mark.parametrize(
    "mutation",
    [
        lambda m: m.update(schema_version=1),
        lambda m: m["implementation"].update(device="a3"),
        lambda m: m["implementation"]["build"].update(abi=None),
        lambda m: m["implementation"]["build"].update(abi="catlass-unary-f32-v1"),
        lambda m: m["implementation"]["build"].update(adapter="catlass"),
        lambda m: m["implementation"]["build"].update(
            specializations=[{"shape": [4], "dtype": "float32"}]
        ),
        lambda m: m["implementation"]["build"]["options"].update(optimization="debug"),
        lambda m: m["implementation"].update(entrypoint="kernel.so:forward"),
        lambda m: m["implementation"]["build"].update(inputs=["kernel.so"]),
    ],
)
def test_unsupported_tensor_contract_combinations_rejected(mutation):
    files = tensor_files()
    manifest = yaml.safe_load(files["candidate/submission.yaml"])
    mutation(manifest)
    files["candidate/submission.yaml"] = yaml.safe_dump(manifest).encode()
    with pytest.raises(SubmissionError):
        validate_tensor_submission(files, "candidate")


def test_tensor_reader_does_not_admit_legacy_sources_and_target_can_describe_triton():
    with pytest.raises(SubmissionError):
        validate_tensor_submission(submission_files(), "candidate")
    assert RequestedTarget(device="a5", language="triton").language == "triton"
    manifest = yaml.safe_load(tensor_files()["candidate/submission.yaml"])
    for abi in (None, "unary-f32-v1", "catlass-unary-f32-v1"):
        altered = copy.deepcopy(manifest)
        altered["schema_version"] = 1
        altered["implementation"]["build"]["abi"] = abi
        with pytest.raises(ValueError):
            SubmissionManifest.model_validate(altered)
