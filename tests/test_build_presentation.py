from uuid import UUID

from sqlalchemy.orm import Session
from test_registration_api import client as client
from test_submission_api import payload, seed

from cannery.artifacts import ArtifactRef
from cannery.builds.state import BuildState, claim_build, finish_build
from cannery.submissions.models import Candidate
from cannery.submissions.state import IntakeState


def test_owned_build_progress_artifact_and_terminal_web_polling(client):
    seed(client)
    created = client.post("/api/v1/kernels/team/identity/submissions", json=payload()).json()
    key = UUID(created["id"])
    engine = client.app.state.test_engine
    with Session(engine) as session, session.begin():
        session.add(
            Candidate(
                id=key,
                binding_id=key,
                binding={},
                manifest={},
                snapshot={},
                content_sha256="a" * 64,
                file_sha256={},
            )
        )
        session.flush()
        session.get(IntakeState, key).status = "awaiting_build"
        session.add(BuildState(id=key))
    info = client.get(created["status_url"]).json()
    assert info["status"] == "build_queued" and info["stages"]["intake"] == "complete"
    assert info["build"] == {"attempts": 0, "artifact": None}
    assert "hx-trigger=" in client.get(f"/submissions/{key}").text
    claim = claim_build(engine, key)
    assert client.get(created["status_url"]).json()["status"] == "building"
    reference = ArtifactRef(
        repository="private", revision="b" * 64, sha256="b" * 64, path="package.tar"
    )
    assert finish_build(engine, claim, artifact=reference)
    info = client.get(created["status_url"]).json()
    assert info["status"] == "awaiting_validation" and info["stages"]["build"] == "complete"
    assert info["stages"]["correctness"] == info["stages"]["benchmark"] == "pending"
    assert info["eta_seconds"] is None and info["build"]["artifact"]["sha256"] == "b" * 64
    assert str(claim.token) not in str(info) and str(claim.attempt) not in str(info)
    assert client.get("/api/v1/submissions/mine").json()[0] == info
    page = client.get(f"/submissions/{key}").text
    assert "hx-trigger=" not in page and "Binary SHA256" in page
