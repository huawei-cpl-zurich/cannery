from types import SimpleNamespace

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session
from test_benchmark_executor import benchmark as benchmark
from test_build_executor import setup_executor as setup_executor
from test_build_state import candidate
from test_numerical_evaluation import arguments as arguments
from test_recipe_spec import recipe_spec
from test_submission_requests import records as records

from cannery.benchmark import scheduling
from cannery.benchmark.evidence import timing_artifact
from cannery.benchmark.state import PerformanceJob, claim_job, finish_job
from cannery.numerics.state import NumericalState, claim_numerical, finish_numerical
from cannery.submissions.models import Candidate


@pytest.fixture
def queue(records, benchmark, monkeypatch):
    engine, _ = records
    _, executor, _, memory, _, spec, suite, cohort = benchmark
    binary = memory.put("binary.tar", b"admitted")

    def retained(engine, identifier, executor):
        with Session(engine) as db:
            frozen = db.get(Candidate, identifier)
        return {
            "candidate": frozen,
            "spec": spec,
            "suite": suite,
            "baseline": "cpu-cache",
            "binary": binary,
            "baseline_artifact": "private-cpu-artifact",
        }

    monkeypatch.setattr(scheduling, "retained_numerical", retained)
    monkeypatch.setattr(
        scheduling,
        "SnapshotBinding",
        SimpleNamespace(
            model_validate=lambda _: SimpleNamespace(source=SimpleNamespace(path="spec"))
        ),
    )
    executor.snapshots = SimpleNamespace(read=lambda _: SimpleNamespace(files=recipe_spec()[0]))

    def add(passed=True):
        key = candidate(records)
        with Session(engine) as db, db.begin():
            db.add(NumericalState(id=key))
        claim = claim_numerical(engine, key)
        finish_numerical(engine, claim, report={"passed": passed})
        return key

    return engine, executor, memory, spec, suite, cohort, add


def test_only_passed_candidates_discovered_once_and_share_reference(queue):
    engine, executor, _, _, _, cohort, add = queue
    first, second, wrong = add(), add(), add(False)
    scheduling.discover(engine, cohort, executor)
    scheduling.discover(engine, cohort, executor)
    with Session(engine) as db:
        jobs = list(db.scalars(select(PerformanceJob)))
        assert len(jobs) == 3
        refs = [r for r in jobs if r.kind == "reference"]
        candidates = [r for r in jobs if r.kind == "candidate"]
        assert len(refs) == 1 and {c.candidate_id for c in candidates} == {first, second}
        assert all(c.reference_key == refs[0].key for c in candidates)
        assert db.get(PerformanceJob, scheduling.candidate_key(wrong)) is None


def test_reference_and_candidate_load_only_matching_completed_cache(queue):
    engine, executor, memory, spec, suite, cohort, add = queue
    key = add()
    scheduling.discover(engine, cohort, executor)
    with Session(engine) as db:
        job = db.get(PerformanceJob, scheduling.candidate_key(key))
        reference = db.get(PerformanceJob, job.reference_key)
    reference_inputs = scheduling.retained(engine, reference, executor)
    assert reference_inputs["kind"] == "reference" and reference_inputs["reference_files"]
    assert "binary" not in reference_inputs
    with pytest.raises(scheduling.ReferencePending):
        scheduling.retained(engine, job, executor)
    values = {c.id: [1.0] * 100 for c in suite.cases[:-1]}
    artifact = memory.put("raw.json", timing_artifact(spec.content_sha256, suite, cohort, values))
    claim = claim_job(engine, reference.key)
    assert finish_job(engine, claim, report={"passed": True}, artifact=artifact)
    inputs = scheduling.retained(engine, job, executor)
    assert inputs["reference_samples"] == values and inputs["reference_files"] is None
    assert inputs["candidate_sha256"] and inputs["binary"]
    assert claim_job(engine, reference.key) is None


def test_failed_reference_blocks_without_dispatching_candidate(queue):
    engine, executor, _, _, _, cohort, add = queue
    key = add()
    scheduling.discover(engine, cohort, executor)
    with Session(engine) as db:
        job = db.get(PerformanceJob, scheduling.candidate_key(key))
    assert finish_job(engine, claim_job(engine, job.reference_key), error="unsupported")
    with pytest.raises(scheduling.BenchmarkBlocked, match="reference_timing_failed"):
        scheduling.retained(engine, job, executor)
