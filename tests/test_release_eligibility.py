import copy
from uuid import uuid4

import pytest
from release_fixtures import admission as admission
from release_fixtures import arguments as arguments
from release_fixtures import benchmark as benchmark
from release_fixtures import records as records
from release_fixtures import setup_executor as setup_executor
from sqlalchemy import update
from sqlalchemy.orm import Session

from cannery.benchmark.state import PerformanceJob, PerformanceResult
from cannery.builds.state import BuildState
from cannery.numerics.state import NumericalResult, NumericalState
from cannery.releases.eligibility import PublicationUnavailable, publication_manifest


def test_completed_evidence_pins_manifest_without_reexecution(admission):
    engine, key, memory, _ = admission
    before = memory.puts
    manifest = publication_manifest(engine, key)
    assert manifest.submission_id == key and manifest.score.geomean_speedup == 0.5
    assert len(manifest.score.cases) == 3 and manifest.score.cases[-1].excluded
    assert manifest.package.sha256 in memory.data and memory.puts == before
    assert manifest.tolerances and all(t.rtol >= 0 and t.atol >= 0 for t in manifest.tolerances)


@pytest.mark.parametrize(
    "model,status",
    [
        (BuildState, "failed"),
        (NumericalState, "failed"),
        (NumericalState, "blocked"),
        (PerformanceJob, "failed"),
        (PerformanceJob, "running"),
    ],
)
def test_incomplete_or_failed_stages_never_publish(admission, model, status):
    engine, key, _, _ = admission
    with engine.begin() as c:
        query = (
            update(model).where(model.candidate_id == key)
            if model is PerformanceJob
            else update(model).where(model.id == key)
        )
        c.execute(query.values(status=status))
    with pytest.raises(PublicationUnavailable):
        publication_manifest(engine, key)


@pytest.mark.parametrize(
    "broken",
    ["identity", "cohort", "missing_case", "mutated_input", "wrong_tolerance", "bad_reference"],
)
def test_mismatched_or_incomplete_retained_evidence_never_publishes(admission, broken):
    engine, key, _, _ = admission
    with Session(engine) as db:
        job = db.query(PerformanceJob).filter_by(candidate_id=key).one()
        identity = copy.deepcopy(job.identity)
        identifier = job.attempt
        reference = job.reference_key
        report = copy.deepcopy(db.get(PerformanceResult, identifier).report)
    if broken == "identity":
        identity["numerical_attempt"] = str(uuid4())
    if broken == "cohort":
        identity["cohort"]["driver"] = "changed"
    if broken == "missing_case":
        report["correctness"][1]["cases"].pop()
    if broken == "mutated_input":
        report["correctness"][0]["cases"][0]["input_unchanged"] = False
    if broken == "wrong_tolerance":
        report["correctness"][0]["cases"][0]["submission_requested"]["atol"] = 99.0
    with engine.begin() as c:
        c.execute(
            update(PerformanceJob)
            .where(PerformanceJob.candidate_id == key)
            .values(identity=identity)
        )
        c.execute(
            update(PerformanceResult)
            .where(PerformanceResult.id == identifier)
            .values(report=report)
        )
        if broken == "bad_reference":
            c.execute(
                update(PerformanceJob)
                .where(PerformanceJob.key == reference)
                .values(status="failed")
            )
    with pytest.raises(PublicationUnavailable):
        publication_manifest(engine, key)


def test_numeric_claim_without_case_proof_never_publishes(admission):
    engine, key, _, _ = admission
    with engine.begin() as c:
        c.execute(
            update(NumericalResult)
            .where(NumericalResult.candidate_id == key)
            .values(report={"passed": True})
        )
    with pytest.raises(PublicationUnavailable):
        publication_manifest(engine, key)
