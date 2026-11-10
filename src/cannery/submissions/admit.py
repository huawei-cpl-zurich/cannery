"""Bounded source-only intake. Never imports or compiles submitted code."""

import json
import sys
from pathlib import Path

from cannery.git_intake import GitSource
from cannery.git_intake.process import IntakeError, attempt_budget
from cannery.git_intake.snapshots import SnapshotStore, extract_tree
from cannery.git_intake.transport import RemoteRule, fetched_source
from cannery.registry.freeze import kernel_key
from cannery.submissions.manifest import SubmissionError, validate_submission


class RepositoryDenied(SubmissionError):
    pass


def main() -> None:
    attempt_budget()
    payload = json.load(sys.stdin)
    source = GitSource.model_validate(payload["source"])
    rules = tuple(RemoteRule.model_validate(rule) for rule in payload["rules"])
    namespace = payload["namespace"]
    if not any(rule.url == source.url and namespace in rule.namespaces for rule in rules):
        raise RepositoryDenied("Repository not authorized")
    scratch = Path(payload["scratch"])
    with fetched_source(source, namespace, rules, scratch=scratch) as repository:
        selected = extract_tree(repository, source)
    manifest = validate_submission(selected.files, source.path).manifest
    if manifest.kernel != payload["kernel"] or manifest.spec_version != payload["spec_version"]:
        raise SubmissionError("Frozen specification mismatch")
    if payload["target"] != {
        "device": manifest.implementation.device,
        "language": manifest.implementation.language,
    }:
        raise SubmissionError("Requested target mismatch")
    name = manifest.kernel.split("/")[1]
    store = SnapshotStore(Path(payload["snapshots"]), scratch=scratch)
    snapshot = store.commit(kernel_key(namespace, name), source, selected)
    print(json.dumps({"snapshot": snapshot.model_dump(mode="json")}))


if __name__ == "__main__":
    try:
        main()
    except RepositoryDenied:
        print(json.dumps({"error": "repository_denied", "retryable": False}))
    except SubmissionError:
        print(json.dumps({"error": "invalid_submission", "retryable": False}))
    except (IntakeError, OSError):
        print(json.dumps({"error": "intake_unavailable", "retryable": True}))
