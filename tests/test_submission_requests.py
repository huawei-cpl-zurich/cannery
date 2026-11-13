from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from cannery.database import Base
from cannery.git_intake import GitSource
from cannery.registry.models import Kernel, SpecVersion
from cannery.submissions.models import Candidate, RequestedTarget, Submission, SubmissionConflict
from cannery.submissions.requests import new_submission, revise_submission


@pytest.fixture
def records(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'requests.db'}")
    Base.metadata.create_all(engine)
    key = uuid4()
    with Session(engine) as session, session.begin():
        session.add(Kernel(id="kernel", namespace="team", name="identity"))
        session.flush()
        session.add(
            SpecVersion(
                id=key,
                kernel_id="kernel",
                version=1,
                content_sha256="a" * 64,
                manifest={"kernel": {"namespace": "team", "name": "identity"}},
                snapshot={},
                file_sha256={},
                smoke={},
            )
        )
    yield engine, key
    engine.dispose()


def initial(records):
    engine, key = records
    source = GitSource(
        url="https://example.com/repo.git",
        commit="a" * 40,
        path="candidate",
        dependencies=("shared",),
    )
    target = RequestedTarget(device="a5", language="catlass")
    return new_submission(engine, "team", key, source, target)


def test_revision_inherits_request_and_does_not_change_parent(records):
    engine, key = records
    parent = initial(records)
    with Session(engine) as session, session.begin():
        session.add(
            Candidate(
                id=parent,
                binding_id=parent,
                binding={},
                manifest={},
                snapshot={},
                content_sha256="b" * 64,
                file_sha256={},
            )
        )
    revision = revise_submission(engine, "team", parent, "b" * 40)
    with Session(engine) as session:
        before, after = session.get(Submission, parent), session.get(Submission, revision)
        assert after.id != before.id and after.parent_id == before.id
        assert before.source["commit"] == "a" * 40 and after.source["commit"] == "b" * 40
        assert after.source | {"commit": before.source["commit"]} == before.source
        assert after.spec_id == before.spec_id == key and after.target == before.target


def test_requests_require_owned_spec_and_retained_parent(records):
    engine, key = records
    source = GitSource(url="https://example.com/repo.git", commit="a" * 40, path="candidate")
    with pytest.raises(SubmissionConflict):
        new_submission(
            engine, "other", key, source, RequestedTarget(device="a5", language="catlass")
        )
    parent = initial(records)
    for namespace, identifier in (("team", parent), ("other", parent), ("team", uuid4())):
        with pytest.raises(SubmissionConflict):
            revise_submission(engine, namespace, identifier, "b" * 40)


@pytest.mark.parametrize("model", [Submission, Candidate])
@pytest.mark.parametrize("operation", ["update", "delete"])
def test_immutable_requests_and_candidates(records, model, operation):
    engine, key = records
    parent = initial(records)
    with Session(engine) as session, session.begin():
        session.add(
            Candidate(
                id=parent,
                binding_id=parent,
                binding={},
                manifest={},
                snapshot={},
                content_sha256="b" * 64,
                file_sha256={},
            )
        )
    with pytest.raises(SubmissionConflict), Session(engine) as session, session.begin():
        row = session.get(model, parent)
        if operation == "delete":
            session.delete(row)
        elif model is Submission:
            row.target = {}
        else:
            row.binding = {"changed": True}


def test_identical_requests_and_revisions_reuse_original_identifier(records):
    engine, _ = records
    first = initial(records)
    assert initial(records) == first
    with Session(engine) as session, session.begin():
        session.add(
            Candidate(
                id=first,
                binding_id=first,
                binding={},
                manifest={},
                snapshot={},
                content_sha256="b" * 64,
                file_sha256={},
            )
        )
    assert revise_submission(engine, "team", first, "a" * 40) == first
    second = revise_submission(engine, "team", first, "b" * 40)
    assert revise_submission(engine, "team", first, "b" * 40) == second
    with Session(engine) as session:
        from sqlalchemy import select

        assert len(list(session.scalars(select(Submission)))) == 2


@pytest.mark.parametrize("field", ["url", "commit", "path", "dependencies", "target"])
def test_request_identity_does_not_cross_source_or_target_boundaries(records, field):
    engine, spec_id = records
    first = initial(records)
    with Session(engine) as session:
        source = dict(session.get(Submission, first).source)
    target = RequestedTarget(device="a5", language="catlass")
    if field == "target":
        target = RequestedTarget(device="a3", language="catlass")
    else:
        source[field] = dict(
            url="https://example.com/other.git",
            commit="c" * 40,
            path="other",
            dependencies=["other"],
        )[field]
    assert (
        new_submission(engine, "team", spec_id, GitSource.model_validate(source), target) != first
    )


def test_revision_and_standalone_requests_keep_separate_binding_contexts(records):
    engine, spec_id = records
    parent = initial(records)
    with Session(engine) as session, session.begin():
        source = dict(session.get(Submission, parent).source)
        session.add(
            Candidate(
                id=parent,
                binding_id=parent,
                binding={"entrypoint": "first"},
                manifest={},
                snapshot={},
                content_sha256="b" * 64,
                file_sha256={},
            )
        )
    source["commit"] = "b" * 40
    standalone = new_submission(
        engine,
        "team",
        spec_id,
        GitSource.model_validate(source),
        RequestedTarget(device="a5", language="catlass"),
    )
    revision = revise_submission(engine, "team", parent, "b" * 40)
    assert revision != standalone
    assert revise_submission(engine, "team", parent, "b" * 40) == revision
    assert (
        new_submission(
            engine,
            "team",
            spec_id,
            GitSource.model_validate(source),
            RequestedTarget(device="a5", language="catlass"),
        )
        == standalone
    )
