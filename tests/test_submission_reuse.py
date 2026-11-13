import copy
from uuid import uuid4

import pytest
from release_fixtures import admission as admission
from release_fixtures import arguments as arguments
from release_fixtures import benchmark as benchmark
from release_fixtures import records as records
from release_fixtures import setup_executor as setup_executor
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from cannery.artifacts import ArtifactError
from cannery.benchmark.state import PerformanceJob
from cannery.builds.coordinator import discover
from cannery.builds.state import BuildState
from cannery.git_intake import GitSource, SnapshotBinding
from cannery.git_intake.snapshots import SelectedTree
from cannery.submissions.api import get_info
from cannery.submissions.manifest import validate_submission
from cannery.submissions.models import Candidate, Submission, SubmissionReuse
from cannery.submissions.reuse import reuse_completed
from cannery.submissions.state import IntakeState


def revision(session, original, snapshots, change=None):
    parent = session.get(Submission, original.id)
    source = GitSource.model_validate(parent.source | {"commit": "b" * 40})
    selected = snapshots.read(SnapshotBinding.model_validate(original.snapshot))
    files, modes = dict(selected.files), dict(selected.modes)
    if change == "source":
        files["candidate/kernel.cpp"] += b"\n// changed"
    elif change == "addition":
        files["candidate/helper.h"], modes["candidate/helper.h"] = b"// new", "100644"
    elif change == "mode":
        modes["candidate/kernel.cpp"] = "100755"
    elif change == "dependency":
        files["shared.h"], modes["shared.h"] = b"// changed dependency", "100644"
        source = source.model_copy(update={"dependencies": ("shared.h",)})
    snapshot = snapshots.commit(
        original.snapshot["internal_repository"], source, SelectedTree(files, modes, "c" * 40)
    )
    valid = validate_submission(files, source.path)
    request = Submission(
        namespace=parent.namespace,
        spec_id=parent.spec_id,
        parent_id=parent.id,
        source=source.model_dump(mode="json"),
        target=parent.target,
    )
    session.add(request)
    session.flush()
    candidate = Candidate(
        id=request.id,
        binding_id=original.binding_id,
        binding=copy.deepcopy(original.binding),
        manifest=valid.manifest.model_dump(mode="json"),
        snapshot=snapshot.model_dump(mode="json"),
        content_sha256=valid.content_sha256,
        file_sha256=valid.file_sha256,
    )
    if change == "binding":
        candidate.binding = {"entrypoint": "different"}
    session.add_all([candidate, IntakeState(id=request.id, status="awaiting_build")])
    session.flush()
    return candidate


def test_unchanged_tree_reuses_verified_results_without_new_execution(admission, setup_executor):
    engine, key, memory, _ = admission
    snapshots = setup_executor[0].snapshots
    before = memory.puts
    with Session(engine) as db, db.begin():
        original = db.get(Candidate, key)
        candidate = revision(db, original, snapshots)
        reuse_completed(engine, db, candidate, snapshots, lambda: memory)
        identifier = candidate.id
    discover(engine)
    with Session(engine) as db:
        assert db.get(SubmissionReuse, identifier).original_id == key
        assert db.get(BuildState, identifier) is None
        info = get_info(db, "team", identifier, publication=True)
        assert info["status"] == "complete" and info["reused_from"] == str(key)
        assert info["source"]["commit"] == "b" * 40
        assert info["benchmark"] == get_info(db, "team", key)["benchmark"]
        assert (
            len(
                list(
                    db.scalars(
                        select(PerformanceJob).where(PerformanceJob.candidate_id == identifier)
                    )
                )
            )
            == 0
        )
    assert memory.puts == before


@pytest.mark.parametrize("change", ["source", "addition", "mode", "dependency", "binding"])
def test_changed_inputs_or_binding_cannot_reuse(admission, setup_executor, change):
    engine, key, memory, _ = admission
    snapshots = setup_executor[0].snapshots
    with Session(engine) as db, db.begin():
        candidate = revision(db, db.get(Candidate, key), snapshots, change)
        reuse_completed(engine, db, candidate, snapshots, lambda: memory)
        assert db.get(SubmissionReuse, candidate.id) is None


def test_failed_measurements_are_not_reused(admission, setup_executor):
    engine, key, memory, _ = admission
    with engine.begin() as connection:
        connection.execute(
            update(PerformanceJob).where(PerformanceJob.candidate_id == key).values(status="failed")
        )
    with Session(engine) as db, db.begin():
        snapshots = setup_executor[0].snapshots
        candidate = revision(db, db.get(Candidate, key), snapshots)
        reuse_completed(engine, db, candidate, snapshots, lambda: memory)
        assert db.get(SubmissionReuse, candidate.id) is None


@pytest.mark.parametrize("failure", ["network", "corrupt"])
def test_unavailable_or_corrupt_artifacts_roll_back_reuse(admission, setup_executor, failure):
    engine, key, memory, _ = admission
    snapshots = setup_executor[0].snapshots
    identifier = uuid4()

    class Faulty:
        def get(self, reference):
            if failure == "network":
                raise ArtifactError("injected disconnect")
            return b"corrupt"

    with pytest.raises((ArtifactError, ValueError)):
        with Session(engine) as db, db.begin():
            candidate = revision(db, db.get(Candidate, key), snapshots)
            identifier = candidate.id
            reuse_completed(engine, db, candidate, snapshots, lambda: Faulty())
    with Session(engine) as db:
        assert db.get(SubmissionReuse, identifier) is None
        assert db.get(Candidate, identifier) is None
