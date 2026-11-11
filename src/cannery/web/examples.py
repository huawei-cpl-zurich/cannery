"""Pinned public examples; filling a form never fetches Git or changes identity."""

REPOSITORY = "https://github.com/huawei-cpl-zurich/helloworld-kernel.git"
COMMIT = "f840604437aec7f42d66bf281680a39ea2a03f04"
NAMESPACE = "team"
SPEC_LABEL = "team/helloworld@spec-v1"
REGISTRATION = {"url": REPOSITORY, "commit": COMMIT, "path": "spec", "name": "helloworld"}
SUBMISSION = {
    "url": REPOSITORY,
    "commit": COMMIT,
    "path": "candidates/helloworld",
    "device": "a5",
    "language": "ascendc",
}

NATIVE_SUBMISSION = {**SUBMISSION, "path": "candidates/helloworld_ascendc"}

CATLASS_SUBMISSION = {**SUBMISSION, "path": "candidates/helloworld_catlass", "language": "catlass"}
