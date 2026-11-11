from html.parser import HTMLParser
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from test_registration_api import client as client
from test_submission_api import seed

from cannery.api import create_app
from cannery.registry.freeze import kernel_key
from cannery.registry.models import Kernel, SpecVersion
from cannery.registry.registrations import Registration
from cannery.settings import Settings

REPO = "https://github.com/huawei-cpl-zurich/helloworld-kernel.git"
SHA = "f840604437aec7f42d66bf281680a39ea2a03f04"


class Fields(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.values = {}
        self.select = None
        self.option = None
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "input" and "name" in attrs:
            self.values[attrs["name"]] = attrs.get("value", "")
        if tag == "select":
            self.select = attrs.get("name")
        if tag == "option":
            self.option = attrs

    def handle_data(self, data):
        if self.select and self.option is not None and data.strip():
            if "selected" in self.option or self.select not in self.values:
                self.values[self.select] = self.option.get("value", data.strip())

    def handle_endtag(self, tag):
        if tag == "option":
            self.option = None
        if tag == "select":
            self.select = None


def test_public_registration_preset_queues_verified_source(client):
    assert Fields(client.get("/register-kernel").text).values["url"] == ""
    page = client.get("/register-kernel?example=helloworld")
    fields = Fields(page.text).values
    assert fields == {
        "url": REPO,
        "commit": SHA,
        "path": "spec",
        "dependencies": "",
        "name": "helloworld",
    }
    response = client.post(
        "/register-kernel",
        data=fields,
        headers={"Origin": "http://testserver"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    key = UUID(response.headers["location"].rsplit("/", 1)[-1])
    with Session(client.app.state.test_engine) as session:
        job = session.get(Registration, key)
        assert job.source == {
            "type": "git",
            "url": REPO,
            "commit": SHA,
            "path": "spec",
            "dependencies": [],
        }
        assert job.namespace == "team" and job.expected_name == "helloworld"


@pytest.mark.parametrize(
    "preset,path,language",
    [
        ("helloworld", "candidates/helloworld", "ascendc"),
        ("helloworld-native", "candidates/helloworld_ascendc", "ascendc"),
        ("helloworld-catlass", "candidates/helloworld_catlass", "catlass"),
        ("helloworld-native-v2", "candidates/helloworld_ascendc_v2", "ascendc"),
        ("helloworld-catlass-v2", "candidates/helloworld_catlass_v2", "catlass"),
    ],
)
def test_toy_preset_selects_required_spec_and_target(client, monkeypatch, preset, path, language):
    from cannery.web.examples import NUMERICAL_COMMIT

    numerical = preset.endswith("-v2")
    pinned = NUMERICAL_COMMIT if numerical else SHA
    engine = client.app.state.test_engine
    with Session(engine) as session, session.begin():
        key = kernel_key("team", "helloworld")
        session.add(Kernel(id=key, namespace="team", name="helloworld"))
        session.flush()
        spec = SpecVersion(
            kernel_id=key,
            version=2 if numerical else 1,
            content_sha256="a" * 64,
            manifest={"kernel": {"namespace": "team", "name": "helloworld"}},
            snapshot={},
            file_sha256={},
            smoke={},
        )
        session.add(spec)
        session.flush()
        expected = str(spec.id)
    seed(client)  # A newer unrelated spec would otherwise be the default option.
    monkeypatch.setattr("cannery.submissions.api.publish_submission", lambda *a: None)
    assert Fields(client.get("/submit-implementation").text).values["url"] == ""
    fields = Fields(client.get(f"/submit-implementation?example={preset}").text).values
    assert fields == {
        "spec_id": expected,
        "device": "a5",
        "language": language,
        "url": REPO,
        "commit": pinned,
        "path": path,
        "dependencies": "",
    }
    response = client.post(
        "/submit-implementation",
        data=fields,
        headers={"Origin": "http://testserver"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    key = response.headers["location"].rsplit("/", 1)[-1]
    job = client.get(f"/api/v1/submissions/{key}").json()
    assert job["kernel"]["name"] == "helloworld" and job["target"]["language"] == language
    assert job["source"]["commit"] == pinned
    assert job["spec_version"] == (2 if numerical else 1)


def test_toy_preset_requires_expected_spec_and_never_changes_namespace(client):
    seed(client)
    page = client.get("/submit-implementation?example=helloworld")
    assert Fields(page.text).values["url"] == ""
    assert "Register the hello-world spec" in page.text
    with TestClient(
        create_app(Settings(dev_identity_enabled=True, dev_namespace="other"))
    ) as other:
        page = other.get("/register-kernel?example=helloworld")
        assert Fields(page.text).values["url"] == ""
        assert "Use hello-world example" not in page.text
        page = other.get("/submit-implementation?example=helloworld")
        assert "Use hello-world toy submission" not in page.text


def test_seeded_registration_preset_uses_v2_pinned_public_source(client):
    from cannery.web.examples import NUMERICAL_COMMIT

    fields = Fields(client.get("/register-kernel?example=helloworld-v2").text).values
    assert fields["path"] == "spec-v2" and fields["commit"] == NUMERICAL_COMMIT
    assert fields["url"] == REPO
