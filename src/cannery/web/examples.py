"""Pinned public examples; filling a form never fetches Git or changes identity."""

REPOSITORY = "https://github.com/huawei-cpl-zurich/helloworld-kernel.git"
COMMIT = "99a8a1a9fd0d1b471e6520bafbbb1020fcdc198d"
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
