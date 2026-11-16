from unittest.mock import MagicMock
from uuid import UUID

import pytest
from sqlalchemy.orm import Session
from test_registration_api import client as client
from test_submission_api import payload, seed

from cannery.submissions.api import SubmitRequest
from cannery.submissions.models import Candidate, Submission


@pytest.fixture(autouse=True)
def dispatch(monkeypatch):
    monkeypatch.setattr("cannery.submissions.api.publish_submission", MagicMock())


@pytest.mark.parametrize(
    "name", ["", "   ", "x" * 81, "x\ny", "x\ty", "x\u200ey", "x\u2028y", "x\u2029y", 123]
)
def test_name_validation(name):
    with pytest.raises(ValueError):
        SubmitRequest.model_validate(payload() | {"implementation_name": name})


def test_named_retries_keep_one_request_and_conflicts_never_dispatch(client, monkeypatch):
    seed(client)
    dispatch = MagicMock()
    monkeypatch.setattr("cannery.submissions.api.publish_submission", dispatch)
    path = "/api/v1/kernels/team/identity/submissions"
    body = payload() | {"implementation_name": "  Tiled 加一  "}
    first = client.post(path, json=body)
    assert first.status_code == 202
    info = client.get(first.json()["status_url"]).json()
    assert info["implementation_name"] == "Tiled 加一" and info["team"] == "team"
    assert client.post(path, json=payload()).json()["id"] == info["id"]
    assert client.post(path, json=body).json()["id"] == info["id"]
    before = dispatch.call_count
    conflict = client.post(path, json=body | {"implementation_name": "Another"})
    assert conflict.status_code == 409 and "fixed implementation name" in conflict.text
    assert dispatch.call_count == before
    assert len(client.get("/api/v1/submissions/mine").json()) == 1


def test_revision_inherits_name_allows_new_name_and_conflicting_retry_fails(client):
    seed(client)
    first = client.post(
        "/api/v1/kernels/team/identity/submissions",
        json=payload() | {"implementation_name": "Initial"},
    ).json()
    parent = UUID(first["id"])
    with Session(client.app.state.test_engine) as db, db.begin():
        db.add(
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
    path = f"/api/v1/submissions/{parent}/revisions"
    inherited = client.post(path, json={"commit": "b" * 40})
    assert inherited.status_code == 202
    assert client.get(inherited.json()["status_url"]).json()["implementation_name"] == "Initial"
    named = client.post(path, json={"commit": "c" * 40, "implementation_name": "New"})
    assert named.status_code == 202
    assert client.get(named.json()["status_url"]).json()["implementation_name"] == "New"
    assert (
        client.post(path, json={"commit": "c" * 40, "implementation_name": "Other"}).status_code
        == 409
    )
    assert (
        client.post(path, json={"commit": "a" * 40, "implementation_name": "Other"}).status_code
        == 409
    )
    with Session(client.app.state.test_engine) as db:
        assert db.get(Submission, parent).implementation_name == "Initial"


def test_unnamed_existing_request_cannot_be_renamed_by_resubmission(client):
    seed(client)
    path = "/api/v1/kernels/team/identity/submissions"
    first = client.post(path, json=payload()).json()
    assert (
        client.get(first["status_url"]).json()["implementation_name"]
        == "Implementation " + first["id"][:8]
    )
    assert client.post(path, json=payload() | {"implementation_name": "New"}).status_code == 409


def test_web_form_accepts_optional_name(client):
    spec = seed(client)
    response = client.post(
        "/submit-implementation",
        data={
            "spec_id": str(spec),
            "device": "a5",
            "language": "catlass",
            "url": "https://example.com/repo.git",
            "commit": "a" * 40,
            "path": "candidate",
            "implementation_name": "Named from browser",
        },
        headers={"Origin": "http://testserver"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    page = client.get(response.headers["location"])
    assert "Named from browser" in page.text
