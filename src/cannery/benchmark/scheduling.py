"""Queue only numerically accepted binaries and bind immutable benchmark inputs."""

import json
from typing import Any
from uuid import UUID

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from cannery.artifacts import ArtifactRef, digest
from cannery.benchmark.contract import Cohort, cache_key, reference_key
from cannery.benchmark.evidence import read_timings
from cannery.benchmark.executor import BenchmarkExecutor
from cannery.benchmark.state import PerformanceJob, PerformanceResult, ensure_job
from cannery.builds.executor import ExecutionFailure
from cannery.builds.state import BuildResult, BuildState
from cannery.git_intake import SnapshotBinding
from cannery.numerics.coordinator import retained as retained_numerical
from cannery.numerics.state import NumericalResult, NumericalState
from cannery.registry.freeze import BaselineEvidence
from cannery.registry.models import SpecVersion
from cannery.submissions.manifest import RequestedValidation, SubmissionManifest
from cannery.submissions.models import Candidate, Submission


class BenchmarkBlocked(ValueError):
    pass


def candidate_key(identifier: UUID) -> str:
    return digest(b"candidate-performance-v1\0" + identifier.bytes)


def discover(engine: Engine, cohort: Cohort) -> None:
    # Discovery relies only on immutable SQL evidence. Blob outages must not stall
    # unrelated queued jobs or prevent their existing deadlines from being enforced.
    with Session(engine) as db:
        rows = list(
            db.execute(
                select(
                    Candidate.id,
                    Submission.spec_id,
                    SpecVersion.content_sha256,
                    SpecVersion.smoke,
                    NumericalState.attempt_id,
                    NumericalResult.report,
                    BuildResult.artifact,
                )
                .join(Submission, Submission.id == Candidate.id)
                .join(SpecVersion, SpecVersion.id == Submission.spec_id)
                .join(NumericalState, NumericalState.id == Candidate.id)
                .join(NumericalResult, NumericalResult.id == NumericalState.attempt_id)
                .join(BuildState, BuildState.id == Candidate.id)
                .join(BuildResult, BuildResult.id == BuildState.attempt_id)
                .where(
                    NumericalState.status == "passed",
                    Candidate.manifest["implementation"]["device"].as_string() == cohort.device,
                    NumericalResult.status == "passed",
                    BuildState.status == "awaiting_validation",
                    BuildResult.status == "complete",
                    ~select(PerformanceJob.key)
                    .where(PerformanceJob.candidate_id == Candidate.id)
                    .exists(),
                )
                .order_by(Candidate.id)
                .limit(50)
            )
        )
    for identifier, spec_id, spec_sha, smoke, attempt, report, binary in rows:
        if not isinstance(identifier, UUID) or not isinstance(report, dict):
            raise ValueError("Invalid retained numerical identity")
        evidence = BaselineEvidence.model_validate(smoke)
        if (
            evidence.spec_sha256 != spec_sha
            or not report
            or not report.get("passed")
            or report.get("spec_sha256") != evidence.spec_sha256
            or report.get("suite_sha256") != evidence.suite_sha256
            or report.get("generator") != evidence.generator
        ):
            raise ValueError("Invalid retained numerical evidence")
        artifact = ArtifactRef.model_validate(binary)
        if artifact.repository != "private" or artifact.revision != artifact.sha256:
            raise ValueError("Invalid admitted artifact")
        identity = {"spec_id": str(spec_id), "cohort": cohort.model_dump(mode="json")}
        reference = reference_key(spec_sha, evidence.suite_sha256, evidence.generator, cohort)
        ensure_job(engine, reference, "reference", identity)
        ensure_job(
            engine,
            candidate_key(identifier),
            "candidate",
            identity
            | {"numerical_attempt": str(attempt), "binary": artifact.model_dump(mode="json")},
            candidate_id=identifier,
            reference_key=reference,
        )


def retained(engine: Engine, job: PerformanceJob, executor: BenchmarkExecutor) -> dict[str, Any]:
    cohort = Cohort.model_validate_json(json.dumps(job.identity["cohort"]))
    if (
        cohort.image != executor.profile.image
        or cohort.runtime != executor.profile.fingerprint
        or cohort.loader_sha256 != executor.wheel_sha256
    ):
        raise BenchmarkBlocked("performance_cohort_changed")
    spec_id = UUID(job.identity["spec_id"])
    with Session(engine) as db:
        if job.kind == "reference":
            identifier = db.scalar(
                select(Candidate.id)
                .join(Submission, Submission.id == Candidate.id)
                .join(NumericalState, NumericalState.id == Candidate.id)
                .where(Submission.spec_id == spec_id, NumericalState.status == "passed")
                .order_by(Candidate.id)
                .limit(1)
            )
        else:
            identifier = job.candidate_id
        if identifier is None:
            raise ExecutionFailure("admitted_candidate_missing")
        numeric = db.get(NumericalState, identifier)
        if numeric is None or numeric.status != "passed":
            raise BenchmarkBlocked("numerical_result_required")
        if job.kind == "candidate" and str(numeric.attempt_id) != job.identity["numerical_attempt"]:
            raise BenchmarkBlocked("numerical_result_changed")
        frozen = db.get(SpecVersion, spec_id)
        request = db.get(Submission, identifier)
        if frozen is None or request is None or request.spec_id != spec_id:
            raise ExecutionFailure("retained_spec_missing")
        binding = SnapshotBinding.model_validate(frozen.snapshot)
    data = retained_numerical(engine, identifier, executor)
    candidate = data.pop("candidate")
    data.pop("baseline_artifact")
    files = executor.snapshots.read(binding).files
    key = cache_key(data["spec"].content_sha256, data["suite"], cohort)
    if job.kind == "reference":
        data.pop("binary")
        if key != job.key:
            raise BenchmarkBlocked("reference_cohort_mismatch")
        defaults = data["spec"].manifest.spec.validation
        requested = RequestedValidation(
            requested_rtol=defaults.default_rtol, requested_atol=defaults.default_atol
        )
    else:
        if (
            job.reference_key != key
            or data["binary"].model_dump(mode="json") != job.identity["binary"]
        ):
            raise BenchmarkBlocked("admitted_binary_changed")
        with Session(engine) as db:
            reference = db.get(PerformanceJob, key)
            if reference is None or reference.status in {"failed", "blocked"}:
                raise BenchmarkBlocked("reference_timing_failed")
            if reference.status != "ready":
                raise ReferencePending("Waiting for reference timing")
            result = db.get(PerformanceResult, reference.attempt)
            if (
                result is None
                or result.status != "ready"
                or result.artifact is None
                or result.report is None
                or not result.report.get("passed")
            ):
                raise BenchmarkBlocked("reference_timing_missing")
            artifact = ArtifactRef.model_validate(result.artifact)
        # Storage outage defers the candidate without ever rerunning a ready reference.
        raw = executor.artifacts.get(artifact)
        try:
            data["reference_samples"] = read_timings(
                raw, artifact, data["spec"].content_sha256, data["suite"], cohort
            )
        except ValueError:
            raise BenchmarkBlocked("reference_timing_invalid") from None
        requested = SubmissionManifest.model_validate(candidate.manifest).validation
        data["candidate_sha256"] = candidate.content_sha256
    return data | {
        "kind": job.kind,
        "cohort": cohort,
        "requested": requested,
        "reference_files": files if job.kind == "reference" else None,
        "reference_path": binding.source.path if job.kind == "reference" else None,
    }


class ReferencePending(Exception):
    pass
