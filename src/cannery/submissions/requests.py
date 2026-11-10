"""Namespace-owned initial requests and commit-only revisions."""

from uuid import UUID

from sqlalchemy import Engine
from sqlalchemy.orm import Session

from cannery.git_intake import GitSource
from cannery.registry.models import SpecVersion
from cannery.submissions.models import Candidate, RequestedTarget, Submission, SubmissionConflict
from cannery.submissions.state import IntakeState


def new_submission(
    engine: Engine, namespace: str, spec_id: UUID, source: GitSource, target: RequestedTarget
) -> UUID:
    with Session(engine, expire_on_commit=False) as session, session.begin():
        spec = session.get(SpecVersion, spec_id)
        if spec is None or spec.manifest["kernel"]["namespace"] != namespace:
            raise SubmissionConflict("Frozen specification not found")
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
