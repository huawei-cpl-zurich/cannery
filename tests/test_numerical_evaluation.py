import json
import struct

import pytest
from test_recipe_spec import recipe_spec

from cannery.artifacts import digest
from cannery.numerics.baseline_format import pack_baseline, unpack_baseline
from cannery.numerics.evaluation import evaluate
from cannery.numerics.recipes import generate, input_sha256, parse_recipes
from cannery.registry.manifest import validate_spec
from cannery.submissions.manifest import RequestedValidation


@pytest.fixture
def arguments():
    files, _, _ = recipe_spec()
    spec = validate_spec(files, "spec")
    suite = parse_recipes(files["spec/recipes.yaml"])
    outputs = {c.id: generate(c.inputs["x"], c.id, "x") for c in suite.cases}
    hashes = {c.id: input_sha256(c.inputs["x"], outputs[c.id]) for c in suite.cases}
    raw = pack_baseline(spec.content_sha256, suite, spec.environment, "sha256:" + "a" * 64, outputs)
    baseline = unpack_baseline(
        raw, digest(raw), spec_sha256=spec.content_sha256, suite=suite, environment=spec.environment
    )
    return (
        baseline,
        suite,
        outputs.copy(),
        hashes,
        spec.manifest.spec.validation,
        RequestedValidation(requested_rtol=0, requested_atol=0),
    )


def test_all_cases_and_empty_outputs_pass_with_both_tolerance_pairs(arguments):
    report = evaluate(*arguments)
    assert report["passed"] and len(report["cases"]) == 3
    assert all(
        c["spec_default"]["passed"] and c["submission_requested"]["passed"] for c in report["cases"]
    )
    assert json.loads(json.dumps(report, allow_nan=False)) == report


@pytest.mark.parametrize("strict", ["spec", "submission"])
def test_both_tolerance_pairs_must_pass_even_when_other_is_loose(arguments, strict):
    baseline, suite, outputs, hashes, defaults, requested = arguments
    first = suite.cases[0].id
    values = [v[0] + 0.125 for v in struct.iter_unpack("<f", outputs[first])]
    outputs[first] = struct.pack("<" + "f" * len(values), *values)
    defaults = defaults.model_copy(
        update={"default_rtol": 0, "default_atol": 0 if strict == "spec" else 1}
    )
    requested = requested.model_copy(update={"requested_atol": 0 if strict == "submission" else 1})
    report = evaluate(baseline, suite, outputs, hashes, defaults, requested)
    case = report["cases"][0]
    assert not report["passed"]
    assert case["spec_default"]["passed"] == (strict != "spec")
    assert case["submission_requested"]["passed"] == (strict != "submission")


def test_input_mutation_fails_even_if_output_is_correct(arguments):
    arguments[3][arguments[1].cases[0].id] = "f" * 64
    report = evaluate(*arguments)
    assert not report["passed"] and not report["cases"][0]["input_unchanged"]


@pytest.mark.parametrize("part", [2, 3])
def test_missing_case_is_not_a_partial_pass(arguments, part):
    arguments[part].pop(arguments[1].cases[0].id)
    with pytest.raises(ValueError, match="mandatory case"):
        evaluate(*arguments)


def test_nan_results_produce_finite_json_and_explicit_failure(arguments):
    first = arguments[1].cases[0].id
    arguments[2][first] = struct.pack("<f", float("nan")) * (len(arguments[2][first]) // 4)
    report = evaluate(*arguments)
    assert not report["passed"]
    assert report["cases"][0]["spec_default"]["nonfinite_mismatches"] > 0
    json.dumps(report, allow_nan=False)
