import copy
from uuid import uuid4

import pytest
from release_fixtures import admission as admission
from release_fixtures import arguments as arguments
from release_fixtures import benchmark as benchmark
from release_fixtures import records as records
from sqlalchemy import update
from sqlalchemy.orm import Session
from test_tensor_publication_numerics import attestation as make_attestation
from test_tensor_publication_numerics import setup_executor as setup_executor
from test_tensor_publication_numerics import tensor_validation as make_validation
from test_tensor_spec_reader import tensor_files
from test_tensor_submission_reader import tensor_files as submission_files

from cannery.benchmark.contract import Cohort, reference_key
from cannery.benchmark.scheduling import candidate_key
from cannery.benchmark.state import PerformanceJob, PerformanceResult
from cannery.git_intake import GitSource
from cannery.git_intake.snapshots import SelectedTree, SnapshotStore
from cannery.numerics.baseline_state import BaselineState
from cannery.numerics.state import NumericalResult, NumericalState
from cannery.registry.models import SpecVersion
from cannery.releases.eligibility import (
    PublicationUnavailable,
    publication_manifest,
    tensor_publication_manifest,
)
from cannery.submissions.manifest import validate_tensor_submission
from cannery.submissions.models import Candidate, Submission


@pytest.fixture
def attestation(tmp_path):
    isolated = tmp_path / "numeric"
    isolated.mkdir()
    setup = setup_executor.__wrapped__(isolated)
    control = {}
    validation = make_validation.__wrapped__(setup, control)
    return make_attestation.__wrapped__(validation, control)


@pytest.fixture
def completed(admission, attestation, tmp_path):
    engine, key, _, _ = admission
    report, evidence, frozen, _, _ = attestation
    snapshots = SnapshotStore(tmp_path / "frozen")
    files = tensor_files()
    binding = snapshots.commit(
        "e" * 64,
        GitSource(url="https://example.com/spec.git", commit="a" * 40, path="spec"),
        SelectedTree(files, dict.fromkeys(files, "100644"), "b" * 40),
    )
    declared = validate_tensor_submission(submission_files(), "candidate")
    with Session(engine) as db:
        job = db.get(PerformanceJob, candidate_key(key))
        old_reference = job.reference_key
        identity = copy.deepcopy(job.identity)
        cohort = Cohort.model_validate(identity["cohort"]).model_copy(
            update={"protocol": "sync-call-tensor-v2"}
        )
        identity["cohort"] = cohort.model_dump(mode="json")
        ref = reference_key(
            frozen.spec.content_sha256, frozen.suite.sha256, evidence.generator, cohort
        )
        spec_id = db.get(Submission, key).spec_id
        attempts = [job.attempt, db.get(PerformanceJob, old_reference).attempt]
        numeric_id = db.get(NumericalState, key).attempt_id
        source = db.get(Candidate, key).snapshot["source"]
    report = copy.deepcopy(report)
    report.update(
        binary_sha256=identity["binary"]["sha256"], candidate_sha256=declared.content_sha256
    )
    reference_check = copy.deepcopy(report)
    for row in reference_check["cases"]:
        for output in row["outputs"].values():
            output["submission_requested"] = copy.deepcopy(output["spec_default"])
    latency = dict(median_us=2.0, p95_us=2.0, samples=100)
    row = dict(id="small", shape=[2, 3], reference=latency, candidate=latency, speedup=1.0)
    with engine.begin() as c:
        c.execute(
            update(SpecVersion)
            .where(SpecVersion.id == spec_id)
            .values(
                content_sha256=frozen.spec.content_sha256,
                file_sha256=frozen.spec.file_sha256,
                manifest=frozen.spec.manifest.model_dump(mode="json"),
                snapshot=binding.model_dump(mode="json"),
                smoke=evidence.model_dump(mode="json"),
            )
        )
        c.execute(
            update(Candidate)
            .where(Candidate.id == key)
            .values(
                manifest=declared.manifest.model_dump(mode="json"),
                content_sha256=declared.content_sha256,
            )
        )
        c.execute(
            update(Submission)
            .where(Submission.id == key)
            .values(
                source=source,
                target={"device": "a5", "language": declared.manifest.implementation.language},
            )
        )
        c.execute(
            update(BaselineState).values(
                key=evidence.baseline_key,
                spec_sha256=frozen.spec.content_sha256,
                artifact=evidence.artifact.model_dump(mode="json"),
            )
        )
        c.execute(update(NumericalState).values(baseline_key=evidence.baseline_key))
        c.execute(
            update(NumericalResult).where(NumericalResult.id == numeric_id).values(report=report)
        )
        c.execute(
            update(PerformanceJob)
            .where(PerformanceJob.key == old_reference)
            .values(key=ref, identity={"spec_id": str(spec_id), "cohort": identity["cohort"]})
        )
        c.execute(
            update(PerformanceJob)
            .where(PerformanceJob.candidate_id == key)
            .values(identity=identity, reference_key=ref)
        )
        for attempt, kind, check in zip(
            attempts, ["candidate", "reference"], [report, reference_check], strict=True
        ):
            scored = row if kind == "candidate" else {"id": "small", "reference": latency}
            result = dict(
                passed=True,
                kind=kind,
                key=ref,
                protocol=cohort.protocol,
                cohort=cohort.model_dump(mode="json"),
                correctness=[check, check],
                cases=[scored],
            )
            if kind == "candidate":
                result["geomean_speedup"] = 1.0
            c.execute(
                update(PerformanceResult)
                .where(PerformanceResult.id == attempt)
                .values(job_key=candidate_key(key) if kind == "candidate" else ref, report=result)
            )
    return engine, key, snapshots


def test_complete_named_mutable_and_integer_evidence_is_read_without_author_import(completed):
    result = tensor_publication_manifest(*completed)
    assert result.cases["small"].aliases == {"updated": "cache"}
    assert result.cases["small"].outputs["indices"].dtype == "int32"
    assert result.score.geomean_speedup == 1.0
    with pytest.raises(PublicationUnavailable):
        publication_manifest(*completed[:2])


@pytest.mark.parametrize(
    "broken",
    ["spec", "binary", "baseline", "candidate", "precheck", "postcheck", "reference", "identity"],
)
def test_mismatched_retained_evidence_cannot_be_published(completed, broken):
    engine, key, _ = completed
    with Session(engine) as db:
        job = db.get(PerformanceJob, candidate_key(key))
        numeric = db.get(NumericalState, key)
        numerical = copy.deepcopy(db.get(NumericalResult, numeric.attempt_id).report)
        result = copy.deepcopy(db.get(PerformanceResult, job.attempt).report)
        attempt, numeric_id, ref = job.attempt, numeric.attempt_id, job.reference_key
        identity = copy.deepcopy(job.identity)
    with engine.begin() as c:
        if broken == "spec":
            c.execute(update(SpecVersion).values(file_sha256={}))
        if broken in {"binary", "baseline", "candidate"}:
            numerical[broken + "_sha256"] = "0" * 64
        if broken in {"precheck", "postcheck"}:
            result["correctness"][int(broken == "postcheck")]["cases"][0]["outputs"]["indices"][
                "spec_default"
            ]["mismatches"] = 1
        if broken == "reference":
            c.execute(
                update(PerformanceJob).where(PerformanceJob.key == ref).values(status="failed")
            )
        if broken == "identity":
            identity["numerical_attempt"] = str(uuid4())
        c.execute(
            update(PerformanceJob)
            .where(PerformanceJob.candidate_id == key)
            .values(identity=identity)
        )
        c.execute(
            update(NumericalResult).where(NumericalResult.id == numeric_id).values(report=numerical)
        )
        c.execute(
            update(PerformanceResult).where(PerformanceResult.id == attempt).values(report=result)
        )
    with pytest.raises(PublicationUnavailable):
        tensor_publication_manifest(*completed)


def test_unavailable_snapshot_fails_closed_then_recovers(completed, monkeypatch):
    snapshots = completed[2]
    read = snapshots.read

    def unavailable(*args):
        raise OSError("injected storage disconnect")

    monkeypatch.setattr(snapshots, "read", unavailable)
    with pytest.raises(PublicationUnavailable):
        tensor_publication_manifest(*completed)
    monkeypatch.setattr(snapshots, "read", read)
    assert tensor_publication_manifest(*completed).submission_id == completed[1]
