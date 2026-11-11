import pytest
from test_freeze import engine as engine
from test_freeze import store as store
from test_recipe_spec import recipe_spec

from cannery.artifacts import ArtifactRef
from cannery.git_intake import GitSource
from cannery.git_intake.snapshots import SelectedTree
from cannery.numerics.baseline_format import baseline_key
from cannery.numerics.baseline_state import claim_baseline, ensure_baseline, finish_baseline
from cannery.numerics.recipes import parse_recipes
from cannery.registry.freeze import BaselineEvidence, SmokeEvidence, freeze_spec, kernel_key
from cannery.registry.manifest import validate_spec
from cannery.registry.models import SpecConflict


def arguments(engine, store, ready=True):
    files, _, _ = recipe_spec()
    spec = validate_spec(files, "spec")
    suite = parse_recipes(files["spec/recipes.yaml"])
    source = GitSource(url="https://example.invalid/spec.git", commit="a" * 40, path="spec")
    binding = store.commit(
        kernel_key("team", "identity"),
        source,
        SelectedTree(files, dict.fromkeys(files, "100644"), "c" * 40),
    )
    key = baseline_key(spec.content_sha256, suite, spec.environment)
    artifact = ArtifactRef(
        repository="private", revision="e" * 64, sha256="e" * 64, path="baseline.tar"
    )
    ensure_baseline(engine, key, spec.content_sha256, binding)
    if ready:
        assert finish_baseline(engine, claim_baseline(engine, key), artifact=artifact)
    evidence = BaselineEvidence(
        spec_sha256=spec.content_sha256,
        image_id="sha256:" + "a" * 64,
        python=spec.environment.python,
        torch=spec.environment.torch,
        cases=len(suite.cases),
        baseline_key=key,
        suite_sha256=suite.sha256,
        generator=suite.generator,
        artifact=artifact,
    )
    return spec, binding, evidence


def freeze(engine, store, args):
    return freeze_spec(engine, *args, store=store, namespace="team", allow_new_version=False)


def test_ready_private_baseline_evidence_freezes_and_reuses_spec(engine, store):
    args = arguments(engine, store)
    frozen = freeze(engine, store, args)
    again = freeze(engine, store, args)
    assert frozen.id == again.id and frozen.smoke["assurance"] == "cached_cpu_reference"
    assert frozen.smoke["artifact"]["repository"] == "private"


def test_queued_cache_cannot_be_used_as_completed_baseline(engine, store):
    with pytest.raises(SpecConflict, match="baseline evidence"):
        freeze(engine, store, arguments(engine, store, ready=False))


@pytest.mark.parametrize(
    "change",
    [
        {"baseline_key": "0" * 64},
        {"suite_sha256": "0" * 64},
        {"cases": 1},
        {"torch": "different"},
        {
            "artifact": ArtifactRef(
                repository="private", revision="f" * 64, sha256="f" * 64, path="other.tar"
            )
        },
    ],
)
def test_other_suite_runtime_case_count_or_output_cannot_freeze_spec(engine, store, change):
    spec, binding, evidence = arguments(engine, store)
    with pytest.raises(SpecConflict):
        freeze(engine, store, (spec, binding, evidence.model_copy(update=change)))


def test_diagnostic_only_evidence_cannot_freeze_recipe_spec(engine, store):
    spec, binding, evidence = arguments(engine, store)
    diagnostic = SmokeEvidence(
        spec_sha256=spec.content_sha256,
        image_id=evidence.image_id,
        python=evidence.python,
        torch=evidence.torch,
        cases=3,
    )
    with pytest.raises(SpecConflict, match="completed CPU baseline"):
        freeze(engine, store, (spec, binding, diagnostic))


def test_unpinned_or_nonprivate_output_reference_is_rejected(engine, store):
    _, _, evidence = arguments(engine, store)
    value = evidence.model_dump(mode="json")
    value["artifact"]["repository"] = "external"
    with pytest.raises(ValueError):
        BaselineEvidence.model_validate(value)
