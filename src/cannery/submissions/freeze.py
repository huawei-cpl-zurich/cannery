"""Retain verified candidates while keeping the original source binding frozen."""

from uuid import UUID

from sqlalchemy.orm import Session

from cannery.git_intake import GitSource, SnapshotBinding
from cannery.git_intake.snapshots import SnapshotStore
from cannery.registry.freeze import kernel_key
from cannery.registry.models import SpecVersion
from cannery.submissions.manifest import validate_submission
from cannery.submissions.models import Candidate, Submission, SubmissionConflict


def freeze_candidate(
    session: Session, identifier: UUID, snapshot: SnapshotBinding, *, store: SnapshotStore
) -> Candidate:
    request = session.get(Submission, identifier)
    if request is None:
        raise SubmissionConflict("Submission request not found")
    try:
        source = GitSource.model_validate(request.source)
        if snapshot.source != source:
            raise SubmissionConflict("Source binding mismatch")
        spec = session.get(SpecVersion, request.spec_id)
        if spec is None or spec.manifest["kernel"]["namespace"] != request.namespace:
            raise SubmissionConflict("Frozen specification not found")
        namespace, name = spec.manifest["kernel"]["namespace"], spec.manifest["kernel"]["name"]
        if snapshot.internal_repository != kernel_key(namespace, name):
            raise SubmissionConflict("Snapshot kernel mismatch")
        validated = validate_submission(store.read(snapshot).files, source.path)
        manifest = validated.manifest
        implementation = manifest.implementation
        if manifest.kernel != f"{namespace}/{name}" or manifest.spec_version != spec.version:
            raise SubmissionConflict("Manifest must target the frozen specification")
        if request.target != {"device": implementation.device, "language": implementation.language}:
            raise SubmissionConflict("Manifest must match the requested device/language")
        identity = {
            "spec_id": str(request.spec_id),
            "repository": source.url,
            "path": source.path,
            "dependencies": list(source.dependencies),
            "device": implementation.device,
            "language": implementation.language,
            "entrypoint": implementation.entrypoint,
            "adapter": implementation.build.adapter,
        }
        if implementation.build.abi is not None:
            identity["abi"] = implementation.build.abi
        binding_id = identifier
        if request.parent_id is not None:
            parent = session.get(Candidate, request.parent_id)
            if parent is None or parent.binding != identity:
                raise SubmissionConflict("Revision must inherit its parent's frozen source binding")
            binding_id = parent.binding_id
        existing = session.get(Candidate, identifier)
        if existing is not None:
            if (
                existing.snapshot != snapshot.model_dump(mode="json")
                or existing.content_sha256 != validated.content_sha256
            ):
                raise SubmissionConflict("Accepted candidate cannot be replaced")
            return existing
        candidate = Candidate(
            id=identifier,
            binding_id=binding_id,
            binding=identity,
            manifest=manifest.model_dump(mode="json"),
            snapshot=snapshot.model_dump(mode="json"),
            content_sha256=validated.content_sha256,
            file_sha256=validated.file_sha256,
        )
        session.add(candidate)
        session.flush()
        return candidate
    except (KeyError, ValueError, TypeError):
        raise SubmissionConflict(
            "Candidate source, target or retained provenance is invalid"
        ) from None
