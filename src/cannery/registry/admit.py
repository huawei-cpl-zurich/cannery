"""Disposable trusted intake attempt; never imports author modules."""

import json
import sys
from pathlib import Path

from cannery.git_intake import GitSource
from cannery.git_intake.process import IntakeError, attempt_budget
from cannery.git_intake.snapshots import SnapshotStore, extract_tree
from cannery.git_intake.transport import RemoteRule, fetched_source
from cannery.registry.freeze import kernel_key
from cannery.registry.manifest import SpecError, validate_spec
from cannery.registry.sandbox import CleanupError, SmokeError, run_smoke


class RepositoryDenied(SpecError):
    pass


def main() -> None:
    attempt_budget()
    payload = json.load(sys.stdin)
    source = GitSource.model_validate(payload["source"])
    namespace = payload["namespace"]
    scratch = Path(payload["scratch"])
    rules = tuple(RemoteRule.model_validate(rule) for rule in payload["rules"])
    if not any(rule.url == source.url and namespace in rule.namespaces for rule in rules):
        raise RepositoryDenied("Repository not authorized")
    with fetched_source(source, namespace, rules, scratch=scratch) as repository:
        selected = extract_tree(repository, source)
    spec = validate_spec(selected.files, source.path)
    if spec.manifest.kernel.namespace != namespace:
        raise SpecError("Namespace mismatch")
    if payload["name"] is not None and spec.manifest.kernel.name != payload["name"]:
        raise SpecError("Kernel name mismatch")
    store = SnapshotStore(Path(payload["snapshots"]), scratch=scratch)
    binding = store.commit(kernel_key(namespace, spec.manifest.kernel.name), source, selected)
    retained = store.read(binding).files
    evidence = run_smoke(
        spec,
        retained,
        source.path,
        image=payload["image"],
        handle=payload["handle"],
        timeout=payload["timeout"],
    )
    print(
        json.dumps(
            {
                "binding": binding.model_dump(mode="json"),
                "evidence": evidence.model_dump(mode="json"),
            }
        )
    )


if __name__ == "__main__":
    try:
        main()
    except RepositoryDenied:
        print(json.dumps({"error": "repository_denied", "retryable": False}))
    except SpecError:
        print(json.dumps({"error": "invalid_spec", "retryable": False}))
    except CleanupError:
        print(json.dumps({"error": "cleanup_pending", "retryable": True}))
    except SmokeError:
        print(json.dumps({"error": "reference_failed", "retryable": True}))
    except (IntakeError, OSError):
        print(json.dumps({"error": "intake_unavailable", "retryable": True}))
