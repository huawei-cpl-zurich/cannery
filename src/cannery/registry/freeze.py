import hashlib

from pydantic import Field
from sqlalchemy import Engine, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from cannery.git_intake import SnapshotBinding
from cannery.registry.manifest import StrictModel, ValidatedSpec
from cannery.registry.models import Kernel, SpecConflict, SpecVersion


class SmokeEvidence(StrictModel):
    image_id: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    python: str
    torch: str
    cases: int = Field(ge=1, le=3)


def kernel_key(namespace: str, name: str) -> str:
    return hashlib.sha256(f"{namespace}/{name}".encode()).hexdigest()


def freeze_in_session(
    session: Session,
    spec: ValidatedSpec,
    binding: SnapshotBinding,
    evidence: SmokeEvidence,
    *,
    namespace: str,
    allow_new_version: bool,
) -> SpecVersion:
    manifest = spec.manifest
    if manifest.kernel.namespace != namespace:
        raise SpecConflict("Specification namespace does not match authorized namespace")
    if evidence.python != spec.environment.python or evidence.torch != spec.environment.torch:
        raise SpecConflict("Smoke runtime differs from locked environment")
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
    evidence: SmokeEvidence,
    *,
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
                namespace=namespace,
                allow_new_version=allow_new_version,
            )
    except IntegrityError:
        raise SpecConflict("Concurrent spec registration; retry with the same source") from None
