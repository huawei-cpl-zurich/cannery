"""Completed admission evidence; remote/device execution remains a CPU fixture."""

import copy
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import update
from sqlalchemy.orm import Session
from test_benchmark_executor import benchmark as benchmark
from test_build_executor import elf
from test_build_executor import setup_executor as setup_executor
from test_build_state import candidate
from test_numerical_evaluation import arguments as arguments
from test_submission_requests import records as records

from cannery.benchmark.contract import cache_key
from cannery.benchmark.scheduling import candidate_key
from cannery.benchmark.state import claim_job, ensure_job, finish_job
from cannery.builds.executor import binary_package
from cannery.builds.state import claim_build, finish_build
from cannery.numerics.baseline_format import pack_baseline
from cannery.numerics.baseline_state import BaselineState
from cannery.numerics.evaluation import evaluate
from cannery.numerics.state import NumericalState, claim_numerical, finish_numerical
from cannery.registry.freeze import BaselineEvidence
from cannery.registry.models import SpecVersion
from cannery.submissions.manifest import RequestedValidation, SubmissionManifest
from cannery.submissions.models import Candidate, Submission


@pytest.fixture
def admission(records, benchmark, setup_executor, arguments, tmp_path):
    engine, spec_id = records
    key = candidate(records)
    run, executor, remote, memory, payload, spec, suite, cohort = benchmark
    original = setup_executor[1]
    declaration = SubmissionManifest.model_validate(original.manifest)
    binary = memory.put(
        "binary.tar", binary_package(elf(), original.content_sha256, executor.profile)
    )
    outputs = arguments[2]
    hashes = arguments[3]
    baseline = arguments[0]
    cpu = memory.put(
        "baseline.tar",
        pack_baseline(spec.content_sha256, suite, spec.environment, "sha256:" + "a" * 64, outputs),
    )
    evidence = BaselineEvidence(
        spec_sha256=spec.content_sha256,
        image_id="sha256:" + "a" * 64,
        python=spec.environment.python,
        torch=spec.environment.torch,
        cases=len(suite.cases),
        baseline_key=baseline.metadata["key"],
        suite_sha256=suite.sha256,
        generator=suite.generator,
        artifact=cpu,
    )
    with engine.begin() as c:
        c.execute(
            update(SpecVersion).values(
                content_sha256=spec.content_sha256,
                manifest=spec.manifest.model_dump(mode="json"),
                smoke=evidence.model_dump(mode="json"),
            )
        )
        c.execute(
            update(Candidate)
            .where(Candidate.id == key)
            .values(
                manifest=original.manifest,
                snapshot=original.snapshot,
                content_sha256=original.content_sha256,
                file_sha256=original.file_sha256,
            )
        )
        c.execute(
            update(Submission)
            .where(Submission.id == key)
            .values(
                source=original.snapshot["source"], target={"device": "a5", "language": "ascendc"}
            )
        )
    with Session(engine) as db, db.begin():
        db.add(
            BaselineState(
                key=evidence.baseline_key,
                spec_sha256=spec.content_sha256,
                snapshot={},
                status="ready",
                attempts=1,
                attempt=uuid4(),
                artifact=cpu.model_dump(mode="json"),
                deadline=datetime.now(UTC) + timedelta(minutes=10),
                due_at=datetime.now(UTC),
            )
        )
        db.add(NumericalState(id=key, baseline_key=evidence.baseline_key))
    assert finish_build(engine, claim_build(engine, key), artifact=binary)
    numeric = claim_numerical(engine, key)
    report = evaluate(
        baseline, suite, outputs, hashes, spec.manifest.spec.validation, declaration.validation
    )
    assert finish_numerical(engine, numeric, report=report)
    reference = cache_key(spec.content_sha256, suite, cohort)
    identity = {"spec_id": str(spec_id), "cohort": cohort.model_dump(mode="json")}
    ensure_job(engine, reference, "reference", identity)
    remote.journal = tmp_path / "release-reference"
    remote.journal.mkdir()
    defaults = spec.manifest.spec.validation
    pair = RequestedValidation(
        requested_rtol=defaults.default_rtol, requested_atol=defaults.default_atol
    )
    ref_report, ref_artifact = run(uuid4(), "reference", requested=pair)
    assert finish_job(
        engine, claim_job(engine, reference), report=ref_report, artifact=ref_artifact
    )
    job = candidate_key(key)
    ensure_job(
        engine,
        job,
        "candidate",
        identity
        | {"numerical_attempt": str(numeric.attempt), "binary": binary.model_dump(mode="json")},
        candidate_id=key,
        reference_key=reference,
    )
    remote.journal = tmp_path / "release-candidate"
    remote.journal.mkdir()
    for case in payload["timing"]["cases"]:
        case["samples_us"] = [4.0] * 100
    actual, artifact = run(
        uuid4(),
        "candidate",
        requested=declaration.validation,
        reference_samples={c.id: [2.0] * 100 for c in suite.cases[:-1]},
    )
    assert finish_job(engine, claim_job(engine, job), report=actual, artifact=artifact)
    return engine, key, memory, copy.deepcopy(actual)
