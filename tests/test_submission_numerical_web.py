"""Submission details render both immutable report formats without changing the API."""

import copy
import struct

import pytest
from test_registration_api import client as client
from test_submission_api import payload, seed

from cannery.numerics.comparison_v2 import compare_tensor
from cannery.web.submissions import numerical_cases


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("passed", [False, True])
def test_numerical_detail_renders_outputs_and_legacy_cases(client, monkeypatch, version, passed):
    seed(client)
    key = client.post("/api/v1/kernels/team/identity/submissions", json=payload()).json()["id"]
    job = client.get(f"/api/v1/submissions/{key}").json()
    comparison = compare_tensor(
        struct.pack("<f", 0),
        struct.pack("<f", 0 if passed else 1),
        "float32",
        rtol=0.001,
        atol=0.001,
    )
    exact = compare_tensor(b"\x01", b"\x01", "int8", rtol=0.001, atol=0.001)
    report = {
        "schema_version": version,
        "cases": [
            {
                "id": "zeros",
                "input_unchanged": passed,
                "spec_default": comparison,
                "submission_requested": comparison,
            }
        ],
    }
    if version == 2:
        report["cases"] = [
            {
                "id": "zeros",
                "readonly_inputs_unchanged": True,
                "mutable_aliases_consistent": passed,
                "outputs": {
                    "value": {"spec_default": comparison, "submission_requested": comparison},
                    "index": {"spec_default": exact, "submission_requested": exact},
                },
            }
        ]
    job.update(candidate={"content_sha256": "a" * 64}, status="failed")
    job["numerical"] = {"status": "passed" if passed else "failed", "attempts": 1, "report": report}
    original = copy.deepcopy(job)
    monkeypatch.setattr("cannery.web.submissions.get_info", lambda *a, **kw: job)
    page = client.get(f"/submissions/{key}")
    assert page.status_code == 200
    assert "Numerical validation results" in page.text
    if version == 2:
        assert "zeros/value" in page.text and "zeros/index" in page.text
        assert "(exact)" in page.text
        assert numerical_cases(job)[0]["input_unchanged"] is passed
    else:
        assert "zeros/value" not in page.text
    assert "undefined" in page.text if not passed else "Max relative error" in page.text
    assert job == original


def test_pending_or_absent_report_has_no_rows():
    assert numerical_cases({"numerical": None}) == []
    assert numerical_cases({"numerical": {"report": None}}) == []
