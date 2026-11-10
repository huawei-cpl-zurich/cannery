"""Pinned public examples; filling a form never fetches Git or changes identity."""

REPOSITORY = "https://github.com/huawei-cpl-zurich/helloworld-kernel.git"
COMMIT = "d838f3fd8e4715388df240c0653b197d7a093c49"
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
