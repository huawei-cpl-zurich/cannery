"""Read-only publication gates over immutable admission and validation evidence."""

import json
import math
from typing import Any
from uuid import UUID

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from cannery.artifacts import ArtifactRef
from cannery.benchmark.contract import Cohort, reference_key
from cannery.benchmark.scheduling import candidate_key
from cannery.benchmark.state import PerformanceJob, PerformanceResult
from cannery.builds.state import BuildResult, BuildState
from cannery.git_intake import SnapshotBinding
from cannery.numerics.baseline_state import BaselineState
from cannery.numerics.comparison import Comparison
from cannery.numerics.state import NumericalResult, NumericalState
from cannery.registry.freeze import BaselineEvidence
from cannery.registry.manifest import Manifest
from cannery.registry.models import SpecVersion
from cannery.releases.contract import ReleaseManifest, Score, Tolerance
from cannery.submissions.manifest import SubmissionManifest
from cannery.submissions.models import Candidate, Submission


class PublicationUnavailable(ValueError):
    pass


def require(condition: Any, code: str = "release_evidence_invalid") -> None:
    if not condition:
        raise PublicationUnavailable(code)


def ready(db: Session, job: PerformanceJob | None, kind: str) -> PerformanceResult:
    require(
        job is not None and job.status == "ready" and job.kind == kind and job.attempt is not None,
        "benchmark_result_required",
    )
    assert job is not None
    result = db.get(PerformanceResult, job.attempt)
    require(
        result is not None
        and result.status == "ready"
        and result.job_key == job.key
        and result.number == job.attempts
        and result.report
        and result.artifact
    )
    assert result is not None
    return result


def check(
    report: dict[str, Any],
    evidence: BaselineEvidence,
    score: Score,
    default: Tolerance,
    requested: Tolerance,
    equal_nan: bool,
) -> None:
    require(
        report["passed"] is True
        and report["schema_version"] == 1
        and report["baseline_key"] == evidence.baseline_key
        and report["spec_sha256"] == evidence.spec_sha256
        and report["suite_sha256"] == evidence.suite_sha256
        and report["generator"] == evidence.generator
    )
    rows = report["cases"]
    require(len(rows) == evidence.cases and [c["id"] for c in rows] == [c.id for c in score.cases])
    for row, case in zip(rows, score.cases, strict=True):
        require(
            row["passed"] is True
            and row["input_unchanged"] is True
            and row["shape"] == list(case.shape)
            and row["dtype"] == "float32"
        )
        for key, pair in [("spec_default", default), ("submission_requested", requested)]:
            data = row[key]
            result = Comparison(**data)
            require(
                result.passed is True
                and result.elements == math.prod(case.shape)
                and result.mismatches == 0
                and result.nonfinite_mismatches == 0
                and result.rtol == pair.rtol
                and result.atol == pair.atol
                and result.equal_nan is equal_nan
            )
            for name in [
                "elements",
                "finite_pairs",
                "mismatches",
                "reference_nonfinite",
                "candidate_nonfinite",
                "nonfinite_mismatches",
                "zero_reference_mismatches",
            ]:
                require(type(data[name]) is int and 0 <= data[name] <= result.elements)
            for name in ["max_absolute_error", "max_relative_error", "rms_error"]:
                value = data[name]
                require(
                    value is None
                    and name == "max_relative_error"
                    or type(value) in (int, float)
                    and math.isfinite(value)
                    and value >= 0
                )


def publication_manifest(engine: Engine, identifier: UUID) -> ReleaseManifest:
    try:
        with Session(engine) as db:
            return retained(db, identifier)
    except PublicationUnavailable:
        raise
    except (KeyError, ValueError, TypeError, AttributeError, OverflowError):
        raise PublicationUnavailable("release_evidence_invalid") from None


def retained(db: Session, identifier: UUID) -> ReleaseManifest:
    sub = db.get(Submission, identifier)
    candidate = db.get(Candidate, identifier)
    build = db.get(BuildState, identifier)
    numeric = db.get(NumericalState, identifier)
    require(
        sub and candidate and build and build.status == "awaiting_validation" and build.attempt_id,
        "admitted_binary_required",
    )
    assert sub and candidate and build
    admitted = db.get(BuildResult, build.attempt_id)
    require(
        admitted and admitted.status == "complete" and admitted.artifact, "admitted_binary_required"
    )
    require(
        numeric and numeric.status == "passed" and numeric.attempt_id, "numerical_result_required"
    )
    assert admitted and numeric
    numerical = db.get(NumericalResult, numeric.attempt_id)
    require(
        numerical
        and numerical.status == "passed"
        and numerical.candidate_id == identifier
        and numerical.number == numeric.attempts
        and numerical.report,
        "numerical_result_required",
    )
    frozen = db.get(SpecVersion, sub.spec_id)
    require(frozen, "retained_spec_required")
    assert frozen
    evidence = BaselineEvidence.model_validate_json(json.dumps(frozen.smoke))
    baseline = db.get(BaselineState, evidence.baseline_key)
    require(
        baseline
        and baseline.status == "ready"
        and baseline.spec_sha256 == frozen.content_sha256
        and evidence.spec_sha256 == frozen.content_sha256
        and baseline.artifact == evidence.artifact.model_dump(mode="json")
        and numeric.baseline_key == evidence.baseline_key,
        "baseline_evidence_required",
    )
    spec = Manifest.model_validate_json(json.dumps(frozen.manifest))
    declaration = SubmissionManifest.model_validate_json(json.dumps(candidate.manifest))
    implementation = declaration.implementation
    require(
        spec.spec.version == frozen.version
        and sub.namespace == spec.kernel.namespace
        and declaration.kernel == f"{spec.kernel.namespace}/{spec.kernel.name}"
        and declaration.spec_version == frozen.version
        and sub.target == {"device": implementation.device, "language": implementation.language}
        and SnapshotBinding.model_validate(candidate.snapshot).source.model_dump(mode="json")
        == sub.source
    )
    job = db.scalar(select(PerformanceJob).where(PerformanceJob.candidate_id == identifier))
    result = ready(db, job, "candidate")
    assert job and result.report and numerical and numerical.report
    reference = db.get(PerformanceJob, job.reference_key)
    base = ready(db, reference, "reference")
    assert reference and base.report
    cohort = Cohort.model_validate_json(json.dumps(job.identity["cohort"]))
    key = reference_key(frozen.content_sha256, evidence.suite_sha256, evidence.generator, cohort)
    package = ArtifactRef.model_validate(admitted.artifact)
    require(
        cohort.device == implementation.device
        and job.key == candidate_key(identifier)
        and job.reference_key == reference.key == key
        and job.identity["spec_id"] == reference.identity["spec_id"] == str(frozen.id)
        and job.identity["numerical_attempt"] == str(numeric.attempt_id)
        and job.identity["binary"] == package.model_dump(mode="json")
        and reference.identity["cohort"] == job.identity["cohort"]
    )
    score = Score.model_validate_json(
        json.dumps({k: result.report[k] for k in ["protocol", "cases", "geomean_speedup"]})
    )
    default = Tolerance(
        rtol=spec.spec.validation.default_rtol, atol=spec.spec.validation.default_atol
    )
    requested = Tolerance(
        rtol=declaration.validation.requested_rtol, atol=declaration.validation.requested_atol
    )
    check(numerical.report, evidence, score, default, requested, spec.spec.validation.equal_nan)
    for report, kind, pair in [
        (result.report, "candidate", requested),
        (base.report, "reference", default),
    ]:
        require(
            report["passed"] is True
            and report["kind"] == kind
            and report["key"] == key
            and report["protocol"] == cohort.protocol
            and report["cohort"] == cohort.model_dump(mode="json")
            and len(report["correctness"]) == 2
        )
        for correctness in report["correctness"]:
            check(correctness, evidence, score, default, pair, spec.spec.validation.equal_nan)
    require([c["id"] for c in base.report["cases"]] == [c.id for c in score.cases])
    for case, row in zip(score.cases, base.report["cases"], strict=True):
        expected: dict[str, Any]
        if case.excluded:
            expected = {"id": case.id, "excluded": "empty_correctness_only"}
        else:
            assert case.reference is not None
            expected = {"id": case.id, "reference": case.reference.model_dump(mode="json")}
        require(row == expected)
    require(implementation.build.abi is not None)
    assert implementation.build.abi is not None and numeric.attempt_id is not None
    pairs = sorted({(default.rtol, default.atol), (requested.rtol, requested.atol)})
    return ReleaseManifest(
        submission_id=identifier,
        spec_id=frozen.id,
        kernel=declaration.kernel,
        spec_version=frozen.version,
        spec_sha256=frozen.content_sha256,
        suite_sha256=evidence.suite_sha256,
        generator=evidence.generator,
        source=SnapshotBinding.model_validate(candidate.snapshot).source,
        source_sha256=candidate.content_sha256,
        device=implementation.device,
        language=implementation.language,
        abi=implementation.build.abi,
        specializations=implementation.build.specializations,
        package=package,
        numerical_attempt=numeric.attempt_id,
        benchmark_attempt=result.id,
        reference_attempt=base.id,
        reference_key=key,
        timing_artifact=ArtifactRef.model_validate(result.artifact),
        reference_artifact=ArtifactRef.model_validate(base.artifact),
        cohort=cohort,
        tolerances=tuple(Tolerance(rtol=r, atol=a) for r, a in pairs),
        score=score,
    )
