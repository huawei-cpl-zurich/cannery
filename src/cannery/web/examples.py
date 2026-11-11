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


NUMERICAL_COMMIT = "c9874e240e9c0f59b90805e06168a3701457d9a4"
NUMERICAL_SPEC_LABEL = "team/helloworld@spec-v2"
NUMERICAL_REGISTRATION = {**REGISTRATION, "commit": NUMERICAL_COMMIT, "path": "spec-v2"}
NUMERICAL_NATIVE = {
    **NATIVE_SUBMISSION,
    "commit": NUMERICAL_COMMIT,
    "path": "candidates/helloworld_ascendc_v2",
}
NUMERICAL_CATLASS = {
    **CATLASS_SUBMISSION,
    "commit": NUMERICAL_COMMIT,
    "path": "candidates/helloworld_catlass_v2",
}
