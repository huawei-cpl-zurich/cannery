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


def test_legacy_canonical_digest_stays_unchanged_with_new_abi():
    import hashlib
    import json

    files = submission_files("ascendc")
    result = validate_submission(files, "candidate")
    manifest = result.manifest.model_dump(mode="json")
    assert "abi" not in manifest["implementation"]["build"]
    expected = hashlib.sha256(
        json.dumps(
            {
                "manifest": manifest,
                "files": {
                    k: v for k, v in result.file_sha256.items() if not k.endswith("submission.yaml")
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    assert result.content_sha256 == expected == (
        "39eefce3783628a2ba850889709d49409a731f8947b0ee77e3210f3325c90299"
    )
    declared = yaml.safe_load(files["candidate/submission.yaml"])
    declared["implementation"]["build"]["abi"] = None
    files["candidate/submission.yaml"] = yaml.safe_dump(declared).encode()
    assert validate_submission(files, "candidate").content_sha256 == expected


def test_explicit_a5_abi_and_trusted_build_plan():
    from cannery.builds.ascendc import AscendCAdapter, BuildError

    files = submission_files("ascendc")
    manifest = yaml.safe_load(files["candidate/submission.yaml"])
    legacy = validate_submission(files, "candidate")
    with pytest.raises(BuildError, match="Unsupported"):
        AscendCAdapter().build(legacy.manifest)
    manifest["implementation"]["device"] = "a5"
    manifest["implementation"]["build"]["abi"] = "unary-f32-v1"
    files["candidate/submission.yaml"] = yaml.safe_dump(manifest).encode()
    native = validate_submission(files, "candidate")
    plan = AscendCAdapter().build(native.manifest)
    assert native.content_sha256 != legacy.content_sha256
    assert 'extern "C" void cannery_launch' in plan.wrapper
    assert plan.command[0] == "bisheng" and "--npu-arch=dav-3510" in plan.command
    manifest["implementation"]["device"] = "a3"
    files["candidate/submission.yaml"] = yaml.safe_dump(manifest).encode()
    with pytest.raises(SubmissionError):
        validate_submission(files, "candidate")
