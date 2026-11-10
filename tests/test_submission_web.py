from uuid import UUID

import pytest
from sqlalchemy.orm import Session
from test_registration_api import client as client
from test_submission_api import payload, seed

from cannery.submissions.models import Candidate
from cannery.submissions.state import IntakeState


@pytest.fixture(autouse=True)
def dispatch(monkeypatch):
    monkeypatch.setattr("cannery.submissions.api.publish_submission", lambda *a: None)


def test_submit_form_and_polling_status(client):
    spec = seed(client)
    assert "team/identity@spec-v1" in client.get("/submit-implementation").text
    response = client.post(
        "/submit-implementation",
        data={
            "spec_id": str(spec),
            "device": "a5",
            "language": "catlass",
            "url": "https://example.com/repo.git",
            "commit": "a" * 40,
            "path": "candidate",
            "dependencies": "shared",
        },
        headers={"Origin": "http://testserver"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    status = client.get(response.headers["location"])
    assert "queued" in status.text and 'hx-trigger="every 2s"' in status.text
    assert "ETA: Unknown" in status.text
    listing = client.get("/my-submissions")
    assert response.headers["location"] in listing.text and "queued" in listing.text


def test_commit_only_web_revision_preserves_parent_and_stops_terminal_polling(client):
    seed(client)
    parent = client.post("/api/v1/kernels/team/identity/submissions", json=payload()).json()["id"]
    with Session(client.app.state.test_engine) as session, session.begin():
        session.add(
            Candidate(
                id=UUID(parent),
                binding_id=UUID(parent),
                binding={},
                manifest={},
                snapshot={},
                content_sha256="b" * 64,
                file_sha256={},
            )
        )
        session.get(IntakeState, UUID(parent)).status = "awaiting_build"
    detail = client.get(f"/submissions/{parent}")
    assert (
        'hx-trigger="every 2s"' not in detail.text
        and "Build, correctness and benchmark are pending" in detail.text
    )
    revision = client.post(
        f"/submissions/{parent}/revisions",
        data={"commit": "b" * 40},
        headers={"Origin": "http://testserver"},
        follow_redirects=False,
    )
    assert revision.status_code == 303
    child_id = revision.headers["location"].split("/")[-1]
    child = client.get(f"/api/v1/submissions/{child_id}").json()
    assert child["source"]["commit"] == "b" * 40 and child["parent_id"] == parent
    assert client.get(f"/api/v1/submissions/{parent}").json()["source"]["commit"] == "a" * 40
    assert (
        client.post(
            f"/submissions/{parent}/revisions",
            data={"commit": "b" * 40, "device": "a3"},
            headers={"Origin": "http://testserver"},
        ).status_code
        == 400
    )


def test_form_boundaries_and_empty_state(client):
    assert "Register a specification" in client.get("/submit-implementation").text
    assert "No submissions yet" in client.get("/my-submissions").text
    assert client.post("/submit-implementation", data={}).status_code == 403
    assert (
        client.post(
            "/submit-implementation", data={}, headers={"Origin": "https://evil.example"}
        ).status_code
        == 403
    )
    response = client.post(
        "/submit-implementation",
        data={},
        headers={
            "Origin": "http://testserver",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    assert response.status_code == 200 and "Choose a frozen specification" in response.text
    response = client.post(
        "/submit-implementation",
        content="x=" + "a" * 17000,
        headers={
            "Origin": "http://testserver",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    assert response.status_code == 413


def test_source_is_escaped_in_status_page(client):
    seed(client)
    body = payload()
    body["source"]["url"] = "https://example.com/<script>alert(1)</script>"
    key = client.post("/api/v1/kernels/team/identity/submissions", json=body).json()["id"]
    response = client.get(f"/submissions/{key}")
    assert "<script>alert" not in response.text and "&lt;script&gt;" in response.text
