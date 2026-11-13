"""Link equivalent retained Git trees to verified historical completed executions."""

from collections.abc import Callable
from typing import Any

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from cannery.artifacts import ArtifactStore
from cannery.benchmark.state import PerformanceJob
from cannery.git_intake import SnapshotBinding
from cannery.git_intake.snapshots import SnapshotStore
from cannery.releases.assets import verify_assets
from cannery.releases.eligibility import publication_manifest
from cannery.submissions.models import Candidate, Submission, SubmissionReuse


def binding_identity(binding: dict[str, Any]) -> dict[str, Any]:
    return binding | {"dependencies": sorted(set(binding.get("dependencies", [])))}


def reuse_completed(
    engine: Engine,
    session: Session,
    candidate: Candidate,
    snapshots: SnapshotStore,
    artifacts: Callable[[], ArtifactStore],
) -> None:
    request = session.get(Submission, candidate.id)
    assert request is not None
    selected = SnapshotBinding.model_validate(candidate.snapshot)
    previous = session.scalars(
        select(Candidate)
        .join(Submission, Submission.id == Candidate.id)
        .join(PerformanceJob, PerformanceJob.candidate_id == Candidate.id)
        .where(
            Submission.namespace == request.namespace,
            Submission.spec_id == request.spec_id,
            Candidate.id != candidate.id,
            Candidate.content_sha256 == candidate.content_sha256,
            Candidate.snapshot["content_sha256"].as_string() == selected.content_sha256,
            PerformanceJob.status == "ready",
            ~select(SubmissionReuse.id).where(SubmissionReuse.id == Candidate.id).exists(),
        )
        .order_by(Submission.created_at, Candidate.id)
    )
    for original in previous:
        if binding_identity(original.binding) != binding_identity(candidate.binding):
            continue
        # Re-read the protected snapshot, validate all saved stage identities, and
        # verify the binary/timing bytes. These checks execute no submitted code.
        snapshots.read(SnapshotBinding.model_validate(original.snapshot))
        manifest = publication_manifest(engine, original.id)
        verify_assets(artifacts(), manifest, lambda: None)
        session.add(SubmissionReuse(id=candidate.id, original_id=original.id))
        return
