import pytest
from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.testclient import TestClient
from release_fixtures import admission as admission
from release_fixtures import arguments as arguments
from release_fixtures import benchmark as benchmark
from release_fixtures import records as records
from release_fixtures import setup_executor as setup_executor

from cannery.releases import state
from cannery.releases.worker import process
from cannery.settings import Settings
from cannery.web import PAGES, ROOT
from cannery.web.submissions import install_submission_web


@pytest.fixture
def web(admission):
    engine, key, store, _ = admission
    app = FastAPI()
    install_submission_web(
        app,
        Settings(
            dev_identity_enabled=True,
            dev_namespace="team",
            release_enabled=True,
            benchmark_enabled=True,
        ),
        engine,
        Jinja2Templates(directory=ROOT / "templates"),
        {"pages": PAGES},
    )
    return engine, key, store, TestClient(app)


def test_explicit_publish_poll_withdraw_republish_preserves_pin(web):
    engine, key, store, client = web
    path = f"/submissions/{key}"
    page = client.get(path)
    assert page.status_code == 200 and "Publish release" in page.text
    assert "Visibility: <strong>private" in page.text
    response = client.post(
        path + "/publish",
        data={},
        headers={
            "Origin": "http://testserver",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    page = client.get(path)
    assert "publishing" in page.text and 'hx-trigger="every 2s"' in page.text
    process(engine, state.due_jobs(engine)[0], store)
    page = client.get(path)
    assert "published" in page.text and 'hx-trigger="every 2s"' not in page.text
    assert "Withdraw publication" in page.text and "Results are private" not in page.text
    assert (
        client.post(
            path + "/withdraw",
            data={},
            headers={
                "Origin": "http://testserver",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            follow_redirects=False,
        ).status_code
        == 303
    )
    assert "Republish release" in client.get(path).text
    assert str(key) in client.get("/my-submissions").text
    assert "withdrawn" in client.get("/my-submissions").text
    before = dict(store.data)
    response = client.post(
        path + "/publish",
        data={},
        headers={
            "Origin": "http://testserver",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    process(engine, state.due_jobs(engine)[0], store)
    assert "published" in client.get(path).text and store.data == before


@pytest.mark.parametrize("action", ["publish", "withdraw"])
def test_forms_reject_cross_origin_and_unexpected_fields(web, action):
    _, key, _, client = web
    path = f"/submissions/{key}/{action}"
    assert client.post(path, data={}).status_code == 403
    assert client.post(path, data={}, headers={"Origin": "http://evil"}).status_code == 403
    assert client.post(path, json={}, headers={"Origin": "http://testserver"}).status_code == 415
    assert (
        client.post(path, data={"force": "1"}, headers={"Origin": "http://testserver"}).status_code
        == 400
    )
