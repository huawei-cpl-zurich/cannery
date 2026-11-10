import pytest
import yaml
from sqlalchemy.orm import Session
from submission_fixtures import submission_files
from test_submission_requests import initial
from test_submission_requests import records as records

from cannery.git_intake import GitSource
from cannery.git_intake.process import git
from cannery.git_intake.snapshots import SelectedTree, SnapshotStore
from cannery.registry.freeze import kernel_key
from cannery.submissions.freeze import freeze_candidate
from cannery.submissions.models import Candidate, Submission, SubmissionConflict
from cannery.submissions.requests import revise_submission


def retained(engine, identifier, store, files=None):
    with Session(engine) as session:
        source = GitSource.model_validate(session.get(Submission, identifier).source)
    files = submission_files() if files is None else files
    files = files | {"shared/helper.py": b"# shared dependency"}
    return store.commit(
        kernel_key("team", "identity"),
        source,
        SelectedTree(files, dict.fromkeys(files, "100644"), "c" * 40),
    )


def test_commit_only_revision_preserves_binding_and_history(records, tmp_path):
    engine, _ = records
    first = initial(records)
    store = SnapshotStore(tmp_path / "snapshots")
    snapshot = retained(engine, first, store)
    with Session(engine) as session, session.begin():
        frozen = freeze_candidate(session, first, snapshot, store=store)
        before = frozen.snapshot.copy(), frozen.manifest.copy(), frozen.content_sha256
        assert freeze_candidate(session, first, snapshot, store=store) is frozen
    second = revise_submission(engine, "team", first, "b" * 40)
    files = submission_files()
    files["candidate/kernel.py"] += b"# optimized revision"
    manifest = yaml.safe_load(files["candidate/submission.yaml"])
    manifest["implementation"]["build"]["options"]["optimization"] = "debug"
    files["candidate/submission.yaml"] = yaml.safe_dump(manifest).encode()
    revised = retained(engine, second, store, files)
    with Session(engine) as session, session.begin():
        next_version = freeze_candidate(session, second, revised, store=store)
        previous = session.get(Candidate, first)
        assert next_version.binding_id == previous.binding_id == first
        assert next_version.binding == previous.binding
        assert next_version.content_sha256 != previous.content_sha256
        assert (previous.snapshot, previous.manifest, previous.content_sha256) == before
    git(store.root / (snapshot.internal_repository + ".git"), "gc", "--prune=now")
    assert (
        store.read(snapshot).files["candidate/kernel.py"]
        != store.read(revised).files["candidate/kernel.py"]
    )


@pytest.mark.parametrize(
    "change",
    [
        lambda m: m.update(kernel="other/identity"),
        lambda m: m.update(spec_version=2),
        lambda m: m["implementation"].update(device="a3"),
    ],
)
def test_manifest_must_match_authorized_request_and_spec(records, tmp_path, change):
    engine, _ = records
    first = initial(records)
    files = submission_files()
    manifest = yaml.safe_load(files["candidate/submission.yaml"])
    change(manifest)
    files["candidate/submission.yaml"] = yaml.safe_dump(manifest).encode()
    store = SnapshotStore(tmp_path / "snapshots")
    snapshot = retained(engine, first, store, files)
    with pytest.raises(SubmissionConflict), Session(engine) as session, session.begin():
        freeze_candidate(session, first, snapshot, store=store)


def test_revision_cannot_change_entrypoint_or_replace_accepted_candidate(records, tmp_path):
    engine, _ = records
    first = initial(records)
    store = SnapshotStore(tmp_path / "snapshots")
    snapshot = retained(engine, first, store)
    with Session(engine) as session, session.begin():
        freeze_candidate(session, first, snapshot, store=store)
    second = revise_submission(engine, "team", first, "b" * 40)
    files = submission_files()
    manifest = yaml.safe_load(files["candidate/submission.yaml"])
    manifest["implementation"]["entrypoint"] = "kernel.py:other"
    files["candidate/submission.yaml"] = yaml.safe_dump(manifest).encode()
    revised = retained(engine, second, store, files)
    with pytest.raises(SubmissionConflict), Session(engine) as session, session.begin():
        freeze_candidate(session, second, revised, store=store)
    with pytest.raises(SubmissionConflict), Session(engine) as session, session.begin():
        freeze_candidate(session, first, revised, store=store)


def test_snapshot_requires_authentic_provenance_and_protected_ref(records, tmp_path):
    engine, _ = records
    first = initial(records)
    store = SnapshotStore(tmp_path / "snapshots")
    snapshot = retained(engine, first, store)
    forged = snapshot.model_copy(update={"external_tree_id": "d" * 40})
    with pytest.raises(SubmissionConflict), Session(engine) as session, session.begin():
        freeze_candidate(session, first, forged, store=store)
    repository = store.root / (snapshot.internal_repository + ".git")
    refs = (
        git(repository, "for-each-ref", "--format=%(refname)", "refs/cannery/snapshots")
        .decode()
        .splitlines()
    )
    for ref in refs:
        git(repository, "update-ref", "-d", ref)
    with pytest.raises(SubmissionConflict), Session(engine) as session, session.begin():
        freeze_candidate(session, first, snapshot, store=store)
