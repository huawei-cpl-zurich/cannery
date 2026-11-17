import copy
from uuid import uuid4

import pytest
from test_build_executor import setup_executor as setup_executor
from test_tensor_numerical_executor import tensor_control as tensor_control
from test_tensor_numerical_executor import tensor_validation as tensor_validation
from test_tensor_submission_reader import tensor_files as submission_files

from cannery.registry.freeze import BaselineEvidence
from cannery.registry.tensor_manifest import ValidatedTensorSpec
from cannery.releases.contract import Tolerance
from cannery.releases.eligibility import PublicationUnavailable
from cannery.releases.tensor_checks import check_tensor
from cannery.submissions.manifest import validate_tensor_submission


@pytest.fixture
def attestation(tensor_validation, tensor_control):
    run, _, _, _, baseline, _ = tensor_validation
    report = run(uuid4())
    options = tensor_control["options"]
    spec, suite, contract = (options[k] for k in ["spec", "suite", "contract"])
    evidence = BaselineEvidence(
        spec_sha256=spec.content_sha256,
        image_id="sha256:" + "a" * 64,
        python=spec.environment.python,
        torch=spec.environment.torch,
        cases=len(suite.cases),
        baseline_key=baseline.metadata["key"],
        suite_sha256=suite.sha256,
        generator=suite.generator,
        artifact=options["baseline_artifact"],
    )
    default = Tolerance(
        rtol=spec.manifest.spec.validation.default_rtol,
        atol=spec.manifest.spec.validation.default_atol,
    )
    declaration = validate_tensor_submission(submission_files(), "candidate").manifest.validation
    requested = Tolerance(rtol=declaration.requested_rtol, atol=declaration.requested_atol)
    return report, evidence, ValidatedTensorSpec(spec, suite, contract), default, requested


def test_complete_controller_attestation_accepts_both_floating_and_exact_outputs(attestation):
    check_tensor(*attestation)
    report = attestation[0]
    assert report["cases"][0]["outputs"]["indices"]["spec_default"]["metric"] == "exact"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(passed=False),
        lambda d: d.update(schema_version=1),
        lambda d: d.update(baseline_key="0" * 64),
        lambda d: d.update(suite_sha256="0" * 64),
        lambda d: d["cases"].clear(),
        lambda d: d["cases"][0].update(passed=False),
        lambda d: d["cases"][0].update(readonly_inputs_unchanged=False),
        lambda d: d["cases"][0].update(mutable_aliases_consistent=False),
        lambda d: d["cases"][0]["inputs"]["x"].update(unchanged=False),
        lambda d: d["cases"][0]["inputs"]["x"].update(mutable=True),
        lambda d: d["cases"][0]["inputs"]["x"].update(sha256="invalid"),
        lambda d: d["cases"][0]["inputs"]["x"].update(data=[1, 2]),
        lambda d: d["cases"][0]["scalars"]["offset"].update(value=True),
        lambda d: d["cases"][0]["outputs"]["updated"].update(alias=None),
        lambda d: d["cases"][0]["outputs"]["updated"].update(shape=[99]),
        lambda d: d["cases"][0]["outputs"]["indices"].update(data=[1, 2]),
        lambda d: d["cases"][0]["outputs"]["indices"]["spec_default"].update(mismatches=1),
        lambda d: d["cases"][0]["outputs"]["indices"]["spec_default"].update(
            max_absolute_error=False
        ),
        lambda d: d["cases"][0]["outputs"]["updated"]["submission_requested"].update(atol=1),
        lambda d: d["cases"][0]["outputs"]["updated"]["spec_default"].update(
            max_absolute_error=float("nan")
        ),
        lambda d: d["cases"][0]["outputs"]["updated"]["spec_default"].update(finite_pairs=True),
    ],
)
def test_incomplete_inconsistent_or_output_bearing_attestation_rejected(attestation, mutate):
    report, *rest = attestation
    report = copy.deepcopy(report)
    mutate(report)
    with pytest.raises(PublicationUnavailable):
        check_tensor(report, *rest)
