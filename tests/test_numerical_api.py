import pytest
from sqlalchemy.orm import Session
from test_build_state import candidate
from test_submission_requests import records as records

from cannery.artifacts import ArtifactRef
from cannery.builds.state import claim_build, finish_build
from cannery.numerics.state import NumericalState, claim_numerical, finish_numerical
from cannery.submissions.api import get_info, list_info


@pytest.mark.parametrize(
    "numeric,public",
    [
        ("queued", "validation_queued"),
        ("validating", "validating"),
        ("passed", "awaiting_benchmark"),
        ("failed", "failed"),
        ("blocked", "awaiting_validation"),
    ],
)
def test_numerical_status_keeps_admission_complete_without_claiming_benchmarks(
    records, numeric, public
):
    engine, _ = records
    key = candidate(records)
    ref = ArtifactRef(repository="private", revision="a" * 64, sha256="a" * 64, path="binary.tar")
    finish_build(engine, claim_build(engine, key), artifact=ref)
    with Session(engine) as db, db.begin():
        db.add(
            NumericalState(
                id=key,
                status="blocked" if numeric == "blocked" else "queued",
                error="recipe_baseline_required" if numeric == "blocked" else None,
            )
        )
    if numeric in ("validating", "passed", "failed"):
        claim = claim_numerical(engine, key)
        if numeric in ("passed", "failed"):
            finish_numerical(engine, claim, report={"passed": numeric == "passed", "cases": []})
    with Session(engine) as db:
        report = get_info(db, "team", key)
        assert report["status"] == public
        assert report["stages"]["build"] == "complete"
        assert report["stages"]["correctness"] == numeric
        assert report["stages"]["benchmark"] == (
            "blocked" if numeric in ("failed", "blocked") else "pending"
        )
        assert report["numerical"]["status"] == numeric
        assert report["build"]["artifact"] == ref.model_dump(mode="json")
        assert list_info(db, "team")[0] == report
        assert list_info(db, "other") == []
        if numeric in ("passed", "failed"):
            assert report["numerical"]["report"]["passed"] == (numeric == "passed")
