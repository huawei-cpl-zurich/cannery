"""Output-free release manifests and runtime-compatible ranked case evidence."""

import json
import math
import statistics
from typing import Annotated, Any, Literal, Self
from uuid import UUID

from pydantic import Field, model_validator

from cannery.artifacts import ArtifactRef, digest
from cannery.benchmark.contract import Cohort
from cannery.git_intake import GitSource
from cannery.registry.manifest import StrictModel, unique_pairs
from cannery.submissions.manifest import KernelPath, Language, Specialization

SHA = r"^[a-f0-9]{64}$"
Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Dimension = Annotated[int, Field(strict=True, ge=0, le=1048576)]


class Tolerance(StrictModel):
    rtol: float = Field(ge=0, allow_inf_nan=False, strict=True)
    atol: float = Field(ge=0, allow_inf_nan=False, strict=True)


class Latency(StrictModel):
    median_us: Positive
    p95_us: Positive
    samples: Literal[100]

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.p95_us < self.median_us:
            raise ValueError("p95 must be at least the median")
        return self


class CaseScore(StrictModel):
    id: str = Field(min_length=1, max_length=128)
    shape: tuple[Dimension, ...] = Field(min_length=1, max_length=4)
    reference: Latency | None = None
    candidate: Latency | None = None
    speedup: Positive | None = None
    excluded: Literal["empty_correctness_only"] | None = None

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if math.prod(self.shape) > 1048576:
            raise ValueError("Case exceeds supported element count")
        if not math.prod(self.shape):
            if self.excluded is None or any(
                x is not None for x in (self.reference, self.candidate, self.speedup)
            ):
                raise ValueError("Empty cases are correctness-only")
        elif (
            self.excluded is not None
            or self.reference is None
            or self.candidate is None
            or self.speedup is None
            or not math.isclose(
                self.speedup,
                self.reference.median_us / self.candidate.median_us,
                rel_tol=1e-12,
                abs_tol=0,
            )
        ):
            raise ValueError("Incomplete or inconsistent case timing")
        return self


class Score(StrictModel):
    protocol: Literal["sync-call-v1"]
    cases: tuple[CaseScore, ...] = Field(min_length=1, max_length=32)
    geomean_speedup: Positive | None

    @model_validator(mode="after")
    def complete(self) -> Self:
        if len({c.id for c in self.cases}) != len(self.cases):
            raise ValueError("Duplicate case IDs")
        ratios = [c.speedup for c in self.cases if c.speedup is not None]
        expected = math.exp(statistics.mean(math.log(r) for r in ratios)) if ratios else None
        if (expected is None) != (self.geomean_speedup is None) or (
            expected is not None
            and self.geomean_speedup is not None
            and not math.isclose(expected, self.geomean_speedup, rel_tol=1e-12, abs_tol=0)
        ):
            raise ValueError("Aggregate differs from mandatory cases")
        return self


class ReleaseManifest(StrictModel):
    schema_version: Literal[1] = 1
    submission_id: UUID
    kernel: KernelPath
    spec_version: int = Field(strict=True, ge=1)
    spec_sha256: str = Field(pattern=SHA)
    suite_sha256: str = Field(pattern=SHA)
    generator: Literal["sha256-f32-v1"]
    source: GitSource
    source_sha256: str = Field(pattern=SHA)
    device: Literal["a5"] = "a5"
    language: Language
    abi: Literal["unary-f32-v1", "catlass-unary-f32-v1"]
    specializations: tuple[Specialization, ...] | None = Field(
        default=None, min_length=1, max_length=4
    )
    package: ArtifactRef
    numerical_attempt: UUID
    benchmark_attempt: UUID
    reference_attempt: UUID
    reference_key: str = Field(pattern=SHA)
    timing_artifact: ArtifactRef
    reference_artifact: ArtifactRef
    cohort: Cohort
    tolerances: tuple[Tolerance, ...] = Field(min_length=1, max_length=2)
    score: Score

    @model_validator(mode="after")
    def bound(self) -> Self:
        for ref in (self.package, self.timing_artifact, self.reference_artifact):
            if ref.repository != "private" or ref.revision != ref.sha256:
                raise ValueError("Release artifacts must be pinned private blobs")
        if self.language == "ascendc":
            if self.abi != "unary-f32-v1" or self.specializations is not None:
                raise ValueError("Invalid Ascend C release ABI")
        elif self.abi != "catlass-unary-f32-v1" or self.specializations is None:
            raise ValueError("CATLASS releases require static specializations")
        if self.specializations is not None:
            shapes = {s.shape for s in self.specializations}
            if len(shapes) != len(self.specializations) or not any(math.prod(s) for s in shapes):
                raise ValueError("Invalid specialization coverage")
            if any(c.shape not in shapes for c in self.score.cases):
                raise ValueError("Mandatory cases outside specialization coverage")
        if len({(t.rtol, t.atol) for t in self.tolerances}) != len(self.tolerances):
            raise ValueError("Duplicate validated tolerances")
        return self

    @property
    def cohort_sha256(self) -> str:
        return digest(canonical(self.cohort.model_dump(mode="json")))

    def encoded(self) -> bytes:
        raw = canonical(self.model_dump(mode="json"))
        if len(raw) > 128 * 1024:
            raise ValueError("Release manifest exceeds budget")
        return raw

    def supports(self, runtime: dict[str, str], shape: tuple[int, ...] | None = None) -> bool:
        return self.cohort.runtime == runtime and (
            shape is None
            or self.specializations is None
            or shape in {s.shape for s in self.specializations}
        )


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def read_manifest(raw: bytes, sha256: str) -> ReleaseManifest:
    if not 0 < len(raw) <= 128 * 1024 or digest(raw) != sha256:
        raise ValueError("Release manifest digest or size mismatch")
    try:
        value = json.loads(raw, object_pairs_hook=unique_pairs)
        manifest = ReleaseManifest.model_validate_json(json.dumps(value))
        if manifest.encoded() != raw:
            raise ValueError("Noncanonical release manifest")
        return manifest
    except (ValueError, TypeError, RecursionError, OverflowError) as error:
        raise ValueError("Invalid release manifest") from error
