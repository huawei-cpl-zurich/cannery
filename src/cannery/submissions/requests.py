"""Namespace-owned initial requests and commit-only revisions."""

from uuid import UUID

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from cannery.git_intake import GitSource
from cannery.registry.models import SpecVersion
from cannery.submissions.models import Candidate, RequestedTarget, Submission, SubmissionConflict
from cannery.submissions.state import IntakeState


def existing_request(
    session: Session,
    namespace: str,
    spec_id: UUID,
    source: GitSource,
    target: RequestedTarget,
    parent_id: UUID | None = None,
) -> UUID | None:
    # The caller holds the specification row lock, including across request insertion.
    requests = session.scalars(
        select(Submission)
        .join(IntakeState, IntakeState.id == Submission.id)
        .where(
            IntakeState.status != "failed",
            Submission.namespace == namespace,
            Submission.parent_id == parent_id,
            Submission.spec_id == spec_id,
            Submission.source["url"].as_string() == source.url,
            Submission.source["commit"].as_string() == source.commit,
            Submission.source["path"].as_string() == source.path,
            Submission.target["device"].as_string() == target.device,
            Submission.target["language"].as_string() == target.language,
        )
        .order_by(Submission.created_at, Submission.id)
    )
    for request in requests:
        if set(request.source.get("dependencies", [])) == set(source.dependencies):
            return request.id
    return None


def new_submission(
    engine: Engine, namespace: str, spec_id: UUID, source: GitSource, target: RequestedTarget
) -> UUID:
    with Session(engine, expire_on_commit=False) as session, session.begin():
        spec = session.get(SpecVersion, spec_id, with_for_update=True)
        if spec is None or spec.manifest["kernel"]["namespace"] != namespace:
            raise SubmissionConflict("Frozen specification not found")
        existing = existing_request(session, namespace, spec_id, source, target)
        if existing is not None:
            return existing
        request = Submission(
            namespace=namespace,
            spec_id=spec_id,
            source=source.model_dump(mode="json"),
            target=target.model_dump(mode="json"),
        )
        session.add(request)
        session.flush()
        session.add(IntakeState(id=request.id))
        return request.id


def revise_submission(engine: Engine, namespace: str, parent_id: UUID, commit: str) -> UUID:
    with Session(engine, expire_on_commit=False) as session, session.begin():
        parent = session.get(Submission, parent_id)
        accepted = session.get(Candidate, parent_id)
        if parent is None or parent.namespace != namespace or accepted is None:
            raise SubmissionConflict("Retained parent submission not found")
        source = GitSource.model_validate(parent.source | {"commit": commit})
        if source == GitSource.model_validate(parent.source):
            return parent.id
        session.get(SpecVersion, parent.spec_id, with_for_update=True)
        existing = existing_request(
            session,
            namespace,
            parent.spec_id,
            source,
            RequestedTarget.model_validate(parent.target),
            parent_id=parent.id,
        )
        if existing is not None:
            return existing
        request = Submission(
            namespace=namespace,
            spec_id=parent.spec_id,
            parent_id=parent.id,
            source=source.model_dump(mode="json"),
            target=parent.target,
        )
        session.add(request)
        session.flush()
        session.add(IntakeState(id=request.id))
        return request.id
