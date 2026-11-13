import copy
import subprocess
from datetime import UTC, datetime
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
from cannery.git_intake.snapshots import SelectedTree, extract_tree
from cannery.releases.api import publish, withdraw_publication
from cannery.settings import Settings
from cannery.submissions.api import get_info
from cannery.submissions.manifest import validate_submission
from cannery.submissions.models import Candidate, RequestedTarget, Submission, SubmissionReuse
from cannery.submissions.requests import new_submission
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
    config = Settings(dev_identity_enabled=True, dev_namespace="team", release_enabled=True)
    publication = publish(config, engine, identifier)
    assert publication["publication_url"] == f"/api/v1/submissions/{key}/publication"
    assert withdraw_publication(config, engine, identifier)["status"] == "withdrawn"
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from cannery.web import install_web

    app = FastAPI()
    install_web(app, config, engine)
    page = TestClient(app).get(f"/submissions/{identifier}")
    assert page.status_code == 200 and "No new execution was performed" in page.text
    assert f'action="/submissions/{key}/publish"' in page.text


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


@pytest.mark.parametrize("change", ["outside", "source", "dependency", "mode", "rename", "add"])
def test_real_git_commit_diff_covers_selected_files_and_dependencies(tmp_path, change):
    repository = tmp_path / "repository"
    repository.mkdir()

    def git(*args):
        return (
            subprocess.check_output(
                ["git", "-C", str(repository), *args], stderr=subprocess.DEVNULL, timeout=10
            )
            .decode()
            .strip()
        )

    git("init")
    git("config", "user.name", "fixture")
    git("config", "user.email", "fixture@local")
    (repository / "candidate").mkdir()
    (repository / "shared").mkdir()
    kernel = repository / "candidate/kernel.cpp"
    kernel.write_text("// kernel")
    (repository / "shared/header.h").write_text("// header")
    (repository / "README.md").write_text("documentation")
    git("add", ".")
    git("commit", "-m", "initial")
    source = GitSource(
        url="https://example.com/repo.git",
        commit=git("rev-parse", "HEAD"),
        path="candidate",
        dependencies=("shared",),
    )
    before = extract_tree(repository / ".git", source)
    if change == "mode":
        kernel.chmod(0o755)
    elif change == "rename":
        kernel.rename(repository / "candidate/renamed.cpp")
    else:
        path = {
            "outside": "README.md",
            "source": "candidate/kernel.cpp",
            "dependency": "shared/header.h",
            "add": "candidate/new.h",
        }[change]
        (repository / path).write_text("changed")
    git("add", ".")
    git("commit", "-m", "revision")
    after = extract_tree(
        repository / ".git", source.model_copy(update={"commit": git("rev-parse", "HEAD")})
    )
    assert ((before.files, before.modes) == (after.files, after.modes)) == (change == "outside")
    if change == "dependency":
        assert before.external_tree_id == after.external_tree_id


def test_historical_duplicates_prefer_completed_execution(admission):
    engine, key, _, _ = admission
    with Session(engine) as db, db.begin():
        original = db.get(Submission, key)
        source, target, spec_id = original.source, original.target, original.spec_id
        older = Submission(
            namespace=original.namespace,
            spec_id=spec_id,
            source=source,
            target=target,
            created_at=datetime(2000, 1, 1, tzinfo=UTC),
        )
        db.add(older)
        db.flush()
        db.add(IntakeState(id=older.id))
    assert (
        new_submission(
            engine,
            "team",
            spec_id,
            GitSource.model_validate(source),
            RequestedTarget.model_validate(target),
        )
        == key
    )
