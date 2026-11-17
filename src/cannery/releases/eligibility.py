"""Read-only publication gates over immutable admission and validation evidence."""

import json
import math
from pathlib import Path
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
from cannery.git_intake.snapshots import SnapshotStore
from cannery.numerics.baseline_state import BaselineState
from cannery.numerics.baseline_v2 import OutputTensorV2
from cannery.numerics.comparison import Comparison
from cannery.numerics.state import NumericalResult, NumericalState
from cannery.registry.freeze import BaselineEvidence
from cannery.registry.manifest import Manifest
from cannery.registry.models import SpecVersion
from cannery.registry.tensor_manifest import ValidatedTensorSpec, validate_tensor_spec
from cannery.releases.contract import CaseScore, ReleaseManifest, Score, Tolerance
from cannery.releases.tensor_contract import (
    TensorCaseScore,
    TensorReleaseCase,
    TensorReleaseManifest,
    TensorScore,
)
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
            result = retained(db, identifier)
            assert isinstance(result, ReleaseManifest)
            return result
    except PublicationUnavailable:
        raise
    except (KeyError, ValueError, TypeError, AttributeError, OverflowError):
        raise PublicationUnavailable("release_evidence_invalid") from None


def tensor_publication_manifest(
    engine: Engine, identifier: UUID, snapshots: SnapshotStore
) -> TensorReleaseManifest:
    """Read immutable evidence without running author code or enabling publication."""
    try:
        with Session(engine) as db:
            sub = db.get(Submission, identifier)
            frozen = db.get(SpecVersion, sub.spec_id) if sub else None
            require(frozen, "retained_spec_required")
            assert frozen
            binding = SnapshotBinding.model_validate(frozen.snapshot)
            tensor = validate_tensor_spec(snapshots.read(binding).files, binding.source.path)
            require(
                tensor.spec.content_sha256 == frozen.content_sha256
                and tensor.spec.file_sha256 == frozen.file_sha256
                and tensor.spec.manifest.model_dump(mode="json") == frozen.manifest
                and tensor.spec.manifest.spec.version == frozen.version,
                "retained_spec_mismatch",
            )
            result = retained(db, identifier, tensor=tensor)
            assert isinstance(result, TensorReleaseManifest)
            return result
    except PublicationUnavailable:
        raise
    except OSError:
        raise PublicationUnavailable("release_snapshot_unavailable") from None
    except (KeyError, ValueError, TypeError, AttributeError, OverflowError):
        raise PublicationUnavailable("release_evidence_invalid") from None


def retained(
    db: Session, identifier: UUID, *, tensor: ValidatedTensorSpec | None = None
) -> ReleaseManifest | TensorReleaseManifest:
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
    is_tensor = evidence.generator == "sha256-tensor-v2"
    require(is_tensor == (tensor is not None), "tensor_publication_not_enabled")
    require(
        (implementation.build.abi == "tensor-v2") == is_tensor
        and (tensor is not None or implementation.language != "triton")
    )
    score_type = TensorScore if is_tensor else Score
    score = score_type.model_validate_json(
        json.dumps({k: result.report[k] for k in ["protocol", "cases", "geomean_speedup"]})
    )
    default = Tolerance(
        rtol=spec.spec.validation.default_rtol, atol=spec.spec.validation.default_atol
    )
    requested = Tolerance(
        rtol=declaration.validation.requested_rtol, atol=declaration.validation.requested_atol
    )

    def numerical_check(report: dict[str, Any], pair: Tolerance) -> None:
        if tensor is not None:
            from cannery.releases.tensor_checks import check_tensor

            check_tensor(report, evidence, tensor, default, pair)
        else:
            assert isinstance(score, Score)
            check(report, evidence, score, default, pair, spec.spec.validation.equal_nan)

    if tensor is not None:
        require([case.id for case in score.cases] == [case.id for case in tensor.suite.cases])
        require(
            numerical.report["binary_sha256"] == package.sha256
            and numerical.report["baseline_sha256"] == evidence.artifact.sha256
            and numerical.report["candidate_sha256"] == candidate.content_sha256
        )
    numerical_check(numerical.report, requested)
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
            numerical_check(correctness, pair)
    require([c["id"] for c in base.report["cases"]] == [c.id for c in score.cases])
    scored_cases: tuple[CaseScore | TensorCaseScore, ...] = score.cases
    for case, row in zip(scored_cases, base.report["cases"], strict=True):
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
    common: dict[str, Any] = dict(
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
    if tensor is not None:
        assert isinstance(score, TensorScore)
        return TensorReleaseManifest(
            **common,
            interface=tensor.spec.manifest.spec.interface,
            cases={
                case.id: TensorReleaseCase(
                    **tensor.contract.cases[case.id].model_dump(),
                    inputs={
                        name: OutputTensorV2(shape=desc.shape, dtype=desc.dtype)
                        for name, desc in case.inputs.items()
                    },
                )
                for case in tensor.suite.cases
            },
        )
    return ReleaseManifest(**common, specializations=implementation.build.specializations)


def publication_document(
    engine: Engine, identifier: UUID, snapshots: SnapshotStore | Path | None = None
) -> ReleaseManifest | TensorReleaseManifest:
    """Dispatch only from the immutable spec, never a caller-selected schema."""
    with Session(engine) as db:
        sub = db.get(Submission, identifier)
        spec = db.get(SpecVersion, sub.spec_id) if sub else None
        tensor = spec is not None and spec.manifest.get("schema_version") == 2
    if tensor:
        if snapshots is None:
            raise PublicationUnavailable("retained_spec_required")
        try:
            store = SnapshotStore(snapshots) if isinstance(snapshots, Path) else snapshots
        except OSError:
            raise PublicationUnavailable("release_snapshot_unavailable") from None
        return tensor_publication_manifest(engine, identifier, store)
    return publication_manifest(engine, identifier)
