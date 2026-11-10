import pytest
import yaml
from submission_fixtures import submission_files

from cannery.submissions.manifest import SubmissionError, validate_submission


@pytest.mark.parametrize("language", ["ascendc", "catlass"])
def test_supported_sources_validate_without_execution(language):
    files = submission_files(language)
    first = validate_submission(files, "candidate")
    assert first.manifest.implementation.build.adapter == language
    assert len(first.content_sha256) == 64 and len(first.file_sha256) == 2
    assert validate_submission(dict(reversed(list(files.items()))), "candidate") == first
    name = next(name for name in files if not name.endswith(".yaml"))
    files[name] += b"// change"
    assert validate_submission(files, "candidate").content_sha256 != first.content_sha256


@pytest.mark.parametrize(
    "mutation",
    [
        lambda m: m.update(schema_version=True),
        lambda m: m.update(kernel="../other"),
        lambda m: m.update(spec_version=0),
        lambda m: m.update(extra="unsupported"),
        lambda m: m["implementation"].update(device="cpu"),
        lambda m: m["implementation"].update(language="triton"),
        lambda m: m["implementation"].update(entrypoint="../kernel.py:forward"),
        lambda m: m["implementation"].update(entrypoint="kernel.py:bad()"),
        lambda m: m["implementation"]["build"].update(adapter="shell"),
        lambda m: m["implementation"]["build"].update(adapter="ascendc"),
        lambda m: m["implementation"]["build"].update(inputs=["kernel.py", "kernel.py"]),
        lambda m: m["implementation"]["build"].update(inputs=["kernel.so"]),
        lambda m: m["implementation"]["build"].update(inputs=[]),
        lambda m: m["implementation"]["build"].update(inputs=["../outside.py"]),
        lambda m: m["implementation"]["build"]["options"].update(command="sh build.sh"),
        lambda m: m["validation"].update(requested_rtol=float("nan")),
        lambda m: m["validation"].update(requested_atol=-1),
    ],
)
def test_unsupported_or_unsafe_manifest_rejected(mutation):
    files = submission_files()
    manifest = yaml.safe_load(files["candidate/submission.yaml"])
    mutation(manifest)
    files["candidate/submission.yaml"] = yaml.safe_dump(manifest).encode()
    with pytest.raises(SubmissionError):
        validate_submission(files, "candidate")


@pytest.mark.parametrize(
    "data", [b"a: 1\na: 2", b"a: &a [*a]", b"!!python/object:os.system {}", b"[a,b]", b"a" * 65537]
)
def test_unsafe_yaml_rejected(data):
    with pytest.raises(SubmissionError):
        validate_submission({"candidate/submission.yaml": data}, "candidate")


def test_missing_inputs_and_unsafe_file_maps_rejected():
    files = submission_files()
    del files["candidate/kernel.py"]
    with pytest.raises(SubmissionError):
        validate_submission(files, "candidate")
    for path in ("../escape", "candidate/.git/config"):
        files = submission_files() | {path: b"bad"}
        with pytest.raises(SubmissionError):
            validate_submission(files, "candidate")


def test_manifest_formatting_does_not_change_digest_but_options_do():
    files = submission_files()
    first = validate_submission(files, "candidate")
    files["candidate/submission.yaml"] += b"\n# cosmetic\n"
    assert validate_submission(files, "candidate").content_sha256 == first.content_sha256
    manifest = yaml.safe_load(files["candidate/submission.yaml"])
    manifest["implementation"]["build"]["options"]["optimization"] = "debug"
    files["candidate/submission.yaml"] = yaml.safe_dump(manifest).encode()
    assert validate_submission(files, "candidate").content_sha256 != first.content_sha256
