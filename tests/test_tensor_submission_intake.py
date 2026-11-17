import pytest
from sqlalchemy import update
from sqlalchemy.orm import Session
from submission_fixtures import submission_files
from test_candidate_freeze import retained
from test_registration_api import client as client
from test_submission_requests import initial
from test_submission_requests import records as records
from test_tensor_submission_reader import tensor_files

from cannery.git_intake.snapshots import SnapshotStore
from cannery.registry.models import SpecVersion
from cannery.submissions.freeze import freeze_candidate
from cannery.submissions.intake import submission_for_spec
from cannery.submissions.manifest import (
    SubmissionError,
    validate_submission,
    validate_tensor_submission,
)
from cannery.submissions.models import Submission, SubmissionConflict


def test_intake_requires_schema_agreement_and_explicit_tensor_opt_in():
    files = tensor_files()
    assert submission_for_spec(
        files, "candidate", 2, tensor_enabled=True
    ) == validate_tensor_submission(files, "candidate")
    for schema, enabled in [(1, True), (1, False), (2, False), (3, True)]:
        with pytest.raises(SubmissionError):
            submission_for_spec(files, "candidate", schema, tensor_enabled=enabled)
    legacy = submission_files()
    for enabled in [False, True]:
        assert submission_for_spec(
            legacy, "candidate", 1, tensor_enabled=enabled
        ) == validate_submission(legacy, "candidate")
    with pytest.raises(SubmissionError):
        submission_for_spec(legacy, "candidate", 2, tensor_enabled=True)


@pytest.mark.parametrize("enabled", [False, True])
def test_retained_tensor_candidate_is_bound_without_author_import(records, tmp_path, enabled):
    engine, _ = records
    key = initial(records)
    with Session(engine) as db:
        spec = db.get(SpecVersion, db.get(Submission, key).spec_id)
    with engine.begin() as c:
        c.execute(update(SpecVersion).values(manifest=spec.manifest | {"schema_version": 2}))
        c.execute(
            update(Submission)
            .where(Submission.id == key)
            .values(target={"device": "a5", "language": "triton"})
        )
    store = SnapshotStore(tmp_path / "snapshots")
    files = tensor_files()
    snapshot = retained(engine, key, store, files)
    with Session(engine) as db, db.begin():
        if enabled:
            result = freeze_candidate(db, key, snapshot, store=store, tensor_enabled=True)
            assert result.manifest["implementation"]["build"]["abi"] == "tensor-v2"
            assert (
                result.content_sha256
                == validate_tensor_submission(
                    files | {"shared/helper.py": b"# shared dependency"}, "candidate"
                ).content_sha256
            )
        else:
            with pytest.raises(SubmissionConflict):
                freeze_candidate(db, key, snapshot, store=store)


@pytest.mark.parametrize(
    "enabled,schema,device,language,status",
    [
        (False, 2, "a5", "triton", 422),
        (True, 2, "a5", "triton", 202),
        (True, 2, "a3", "ascendc", 422),
        (True, 1, "a5", "triton", 422),
        (True, 2, "a5", "catlass", 202),
        (True, 2, "a5", "ascendc", 202),
    ],
)
def test_public_api_gates_tensor_schema_and_a5_target(
    client, monkeypatch, enabled, schema, device, language, status
):
    from fastapi.testclient import TestClient
    from test_submission_api import payload, seed

    from cannery.api import create_app
    from cannery.settings import Settings

    engine = client.app.state.test_engine
    spec_id = seed(client)
    with Session(engine) as db:
        before = db.get(SpecVersion, spec_id).manifest
    with engine.begin() as c:
        c.execute(
            update(SpecVersion)
            .where(SpecVersion.id == spec_id)
            .values(manifest=before | {"schema_version": schema})
        )
    monkeypatch.setattr("cannery.api.make_engine", lambda _: engine)
    monkeypatch.setattr("cannery.submissions.api.publish_submission", lambda *args: None)
    with TestClient(create_app(Settings(dev_identity_enabled=True, tensor_enabled=enabled))) as api:
        result = api.post(
            "/api/v1/kernels/team/identity/submissions",
            json=payload() | {"language": language, "device": device},
        )
        assert result.status_code == status, result.text
