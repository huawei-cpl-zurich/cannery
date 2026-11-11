import hashlib
from typing import Literal

from pydantic import Field, model_validator
from sqlalchemy import Engine, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from cannery.artifacts import ArtifactRef
from cannery.git_intake import SnapshotBinding
from cannery.git_intake.process import IntakeError
from cannery.git_intake.snapshots import SnapshotStore
from cannery.numerics.baseline_format import baseline_key
from cannery.numerics.baseline_state import BaselineState
from cannery.numerics.recipes import parse_recipes
from cannery.registry.manifest import (
    RecipeEntrypoint,
    SpecError,
    StrictModel,
    ValidatedSpec,
    validate_spec,
)
from cannery.registry.models import Kernel, SpecConflict, SpecVersion


class SmokeEvidence(StrictModel):
    """Bounded program observations; never independent proof of Python execution."""

    assurance: Literal["diagnostic"] = "diagnostic"
    spec_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    image_id: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    python: str
    torch: str
    cases: int = Field(ge=1, le=3)


class BaselineEvidence(StrictModel):
    """Canonical CPU output cache completed before a recipe-based spec freezes."""

    assurance: Literal["cached_cpu_reference"] = "cached_cpu_reference"
    spec_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    image_id: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    python: str
    torch: str
    cases: int = Field(ge=1, le=32)
    baseline_key: str = Field(pattern=r"^[a-f0-9]{64}$")
    suite_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    generator: Literal["sha256-f32-v1"]
    artifact: ArtifactRef

    @model_validator(mode="after")
    def private_outputs(self) -> "BaselineEvidence":
        if self.artifact.repository != "private" or self.artifact.revision != self.artifact.sha256:
            raise ValueError("CPU baseline requires pinned private output storage")
        return self


def kernel_key(namespace: str, name: str) -> str:
    return hashlib.sha256(f"{namespace}/{name}".encode()).hexdigest()


def freeze_in_session(
    session: Session,
    spec: ValidatedSpec,
    binding: SnapshotBinding,
    evidence: SmokeEvidence | BaselineEvidence,
    *,
    store: SnapshotStore,
    namespace: str,
    allow_new_version: bool,
) -> SpecVersion:
    try:
        files = store.read(binding).files
        retained = validate_spec(files, binding.source.path)
    except (IntakeError, SpecError):
        raise SpecConflict("Retained snapshot verification failed") from None
    if retained != spec or evidence.spec_sha256 != retained.content_sha256:
        raise SpecConflict("Validated content and smoke evidence must match the retained snapshot")
    manifest = spec.manifest
    if manifest.kernel.namespace != namespace:
        raise SpecConflict("Specification namespace does not match authorized namespace")
    if evidence.python != spec.environment.python or evidence.torch != spec.environment.torch:
        raise SpecConflict("Smoke runtime differs from locked environment")
    if isinstance(manifest.spec.cases, RecipeEntrypoint):
        prefix = "" if binding.source.path == "." else binding.source.path + "/"
        suite = parse_recipes(files[prefix + manifest.spec.cases.recipe_file])
        if not isinstance(evidence, BaselineEvidence):
            raise SpecConflict("Recipe-based specs require a completed CPU baseline")
        row = session.get(BaselineState, evidence.baseline_key)
        if (
            evidence.baseline_key != baseline_key(spec.content_sha256, suite, spec.environment)
            or evidence.suite_sha256 != suite.sha256
            or evidence.generator != suite.generator
            or evidence.cases != len(suite.cases)
            or row is None
            or row.status != "ready"
            or row.spec_sha256 != spec.content_sha256
            or row.artifact != evidence.artifact.model_dump(mode="json")
        ):
            raise SpecConflict("CPU baseline evidence does not match the frozen recipe suite")
    elif not isinstance(evidence, SmokeEvidence):
        raise SpecConflict("Python-generator specs require diagnostic smoke evidence")
    key = kernel_key(namespace, manifest.kernel.name)
    if binding.internal_repository != key:
        raise SpecConflict("Snapshot belongs to another kernel")
    kernel = session.get(Kernel, key, with_for_update=True)
    if kernel is None:
        session.add(Kernel(id=key, namespace=namespace, name=manifest.kernel.name))
        session.flush()
    existing = session.scalar(
        select(SpecVersion).where(
            SpecVersion.kernel_id == key, SpecVersion.content_sha256 == spec.content_sha256
        )
    )
    if existing is not None:
        if existing.version != manifest.spec.version:
            raise SpecConflict("Unchanged content must retain its existing spec version")
        return existing
    latest = (
        session.scalar(select(func.max(SpecVersion.version)).where(SpecVersion.kernel_id == key))
        or 0
    )
    if latest and not allow_new_version:
        raise SpecConflict("Creating a new version requires administrator authorization")
    if manifest.spec.version != latest + 1:
        raise SpecConflict("Spec versions must be registered sequentially from version 1")
    frozen = SpecVersion(
        kernel_id=key,
        version=manifest.spec.version,
        content_sha256=spec.content_sha256,
        manifest=manifest.model_dump(mode="json"),
        snapshot=binding.model_dump(mode="json"),
        file_sha256=spec.file_sha256,
        smoke=evidence.model_dump(mode="json"),
    )
    session.add(frozen)
    session.flush()
    return frozen


def freeze_spec(
    engine: Engine,
    spec: ValidatedSpec,
    binding: SnapshotBinding,
    evidence: SmokeEvidence | BaselineEvidence,
    *,
    store: SnapshotStore,
    namespace: str,
    allow_new_version: bool,
) -> SpecVersion:
    try:
        with Session(engine, expire_on_commit=False) as session, session.begin():
            return freeze_in_session(
                session,
                spec,
                binding,
                evidence,
                store=store,
                namespace=namespace,
                allow_new_version=allow_new_version,
            )
    except IntegrityError:
        raise SpecConflict("Concurrent spec registration; retry with the same source") from None
