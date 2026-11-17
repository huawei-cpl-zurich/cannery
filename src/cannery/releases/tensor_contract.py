"""Output-free tensor release reader; publication admission remains separate."""

import json
from typing import Literal, Self
from uuid import UUID

from pydantic import Field, model_validator

from cannery.artifacts import ArtifactRef, digest
from cannery.benchmark.contract import Cohort, reference_key
from cannery.git_intake import GitSource
from cannery.numerics.baseline_v2 import OutputTensorV2
from cannery.numerics.recipes_v2 import RecipeSuiteV2
from cannery.numerics.tensor_contract import TensorCaseV2, TensorContractV2
from cannery.registry.manifest import Interface, Name, StrictModel, unique_pairs
from cannery.releases.contract import SHA, CaseScore, Latency, Positive, Score, Tolerance, canonical
from cannery.submissions.manifest import KernelPath, TensorLanguage


class TensorReleaseCase(TensorCaseV2):
    inputs: dict[Name, OutputTensorV2] = Field(min_length=1, max_length=32)


class TensorCaseScore(StrictModel):
    id: Name
    shape: tuple[int, ...] = Field(min_length=1, max_length=4)
    reference: Latency | None = None
    candidate: Latency | None = None
    speedup: Positive | None = None
    excluded: Literal["empty_correctness_only"] | None = None


class TensorScore(StrictModel):
    protocol: Literal["sync-call-tensor-v2"]
    cases: tuple[TensorCaseScore, ...] = Field(min_length=1, max_length=32)
    geomean_speedup: Positive | None

    @model_validator(mode="after")
    def complete(self) -> Self:
        # Reuse strict latency/ratio/geomean checks. The release binds emptiness
        # to all named tensors, rather than the first input's display shape.
        Score(
            protocol="sync-call-v1",
            cases=tuple(
                CaseScore.model_validate(
                    case.model_dump() | {"shape": (0,) if case.excluded else (1,)}
                )
                for case in self.cases
            ),
            geomean_speedup=self.geomean_speedup,
        )
        return self


class TensorReleaseManifest(StrictModel):
    schema_version: Literal[2] = 2
    submission_id: UUID
    kernel: KernelPath
    spec_id: UUID
    spec_version: int = Field(strict=True, ge=1)
    spec_sha256: str = Field(pattern=SHA)
    suite_sha256: str = Field(pattern=SHA)
    generator: Literal["sha256-tensor-v2"]
    source: GitSource
    source_sha256: str = Field(pattern=SHA)
    device: Literal["a5"] = "a5"
    language: TensorLanguage
    abi: Literal["tensor-v2"]
    interface: Interface
    cases: dict[Name, TensorReleaseCase] = Field(min_length=1, max_length=32)
    package: ArtifactRef
    numerical_attempt: UUID
    benchmark_attempt: UUID
    reference_attempt: UUID
    reference_key: str = Field(pattern=SHA)
    timing_artifact: ArtifactRef
    reference_artifact: ArtifactRef
    cohort: Cohort
    tolerances: tuple[Tolerance, ...] = Field(min_length=1, max_length=2)
    score: TensorScore

    @model_validator(mode="after")
    def bound(self) -> Self:
        for ref in (self.package, self.timing_artifact, self.reference_artifact):
            if ref.repository != "private" or ref.revision != ref.sha256:
                raise ValueError("Release artifacts must be pinned private blobs")
        if self.cohort.device != "a5" or self.cohort.protocol != self.score.protocol:
            raise ValueError("Tensor release requires the A5 tensor timing cohort")
        if self.reference_key != reference_key(
            self.spec_sha256, self.suite_sha256, self.generator, self.cohort
        ):
            raise ValueError("Reference key differs from the frozen tensor cohort")
        if len({(t.rtol, t.atol) for t in self.tolerances}) != len(self.tolerances):
            raise ValueError("Duplicate validated tolerances")
        if set(self.cases) != {c.id for c in self.score.cases}:
            raise ValueError("Score must cover every mandatory tensor case")
        # Zero recipes validate only descriptors/dispatch domains; no inputs or
        # oracle outputs are generated, stored or exposed by this reader.
        suite = RecipeSuiteV2.model_validate(
            {
                "schema_version": 2,
                "generator": self.generator,
                "cases": [
                    {
                        "id": name,
                        "inputs": {
                            key: tensor.model_dump() | {"distribution": "zeros", "seed": 0}
                            for key, tensor in case.inputs.items()
                        },
                    }
                    for name, case in self.cases.items()
                ],
            }
        )
        contract = TensorContractV2(
            schema_version=2,
            cases={
                name: TensorCaseV2.model_validate(case.model_dump(exclude={"inputs"}))
                for name, case in self.cases.items()
            },
        )
        contract.validate_suite(suite)
        for scored in self.score.cases:
            case = self.cases[scored.id]
            if set(case.inputs) != set(self.interface.inputs) or set(case.outputs) != set(
                self.interface.outputs
            ):
                raise ValueError("Tensor release differs from its ordered interface")
            if scored.shape != case.inputs[self.interface.inputs[0]].shape:
                raise ValueError("Display shape differs from the first ordered input")
            timed = any(t.nbytes for t in (*case.inputs.values(), *case.outputs.values()))
            if timed == (scored.excluded is not None):
                raise ValueError("Only entirely empty tensor cases may be excluded")
        return self

    @property
    def cohort_sha256(self) -> str:
        return digest(json.dumps(self.cohort.model_dump(mode="json"), sort_keys=True).encode())

    def encoded(self) -> bytes:
        raw = canonical(self.model_dump(mode="json"))
        if len(raw) > 256 * 1024:
            raise ValueError("Tensor release manifest exceeds budget")
        return raw


def read_tensor_manifest(raw: bytes, sha256: str) -> TensorReleaseManifest:
    if type(raw) is not bytes or not 0 < len(raw) <= 256 * 1024 or digest(raw) != sha256:
        raise ValueError("Tensor release manifest digest or size mismatch")
    try:
        value = json.loads(raw, object_pairs_hook=unique_pairs)
        manifest = TensorReleaseManifest.model_validate_json(json.dumps(value))
        if manifest.encoded() != raw:
            raise ValueError("Noncanonical tensor release manifest")
        return manifest
    except (ValueError, TypeError, RecursionError, OverflowError) as error:
        raise ValueError("Invalid tensor release manifest") from error
