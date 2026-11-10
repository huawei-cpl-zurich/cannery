import json
from unittest.mock import Mock

import httpx
from typer.testing import CliRunner

from cannery.cli import app
from cannery.submissions import cli

Client = httpx.Client
KEY = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"


def invoke(monkeypatch, replies, args):
    requests = []

    def handle(request):
        requests.append(request)
        status, body = replies.pop(0)
        return httpx.Response(status, json=body)

    client = Client(transport=httpx.MockTransport(handle))
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client)
    monkeypatch.setattr(cli, "sleep", lambda _: None)
    return CliRunner().invoke(app, args), requests


def arguments():
    return [
        "submit",
        "team/identity",
        "--repo",
        "https://example.com/repo.git",
        "--commit",
        "a" * 40,
        "--path",
        "candidate/",
        "--spec-version",
        "1",
        "--device",
        "a5",
        "--lang",
        "catlass",
    ]


def test_submit_wait_recovers_transient_failure_without_claiming_build(monkeypatch):
    replies = [
        (202, {"id": KEY}),
        (503, {}),
        (200, {"status": "intaking"}),
        (200, {"status": "awaiting_build"}),
    ]
    result, requests = invoke(monkeypatch, replies, arguments() + ["--wait"])
    assert (
        result.exit_code == 0
        and "awaiting build" in result.output
        and "ETA unknown" in result.output
    )
    body = json.loads(requests[0].content)
    assert body["source"]["path"] == "candidate" and body["spec_version"] == 1
    assert body["device"] == "a5" and body["language"] == "catlass"


def test_revision_sends_only_new_commit(monkeypatch):
    result, requests = invoke(
        monkeypatch, [(202, {"id": KEY})], ["revise", KEY, "--commit", "b" * 40]
    )
    assert result.exit_code == 0 and json.loads(requests[0].content) == {"commit": "b" * 40}
    assert requests[0].url.path == f"/api/v1/submissions/{KEY}/revisions"


def test_failed_or_denied_status_exits_nonzero(monkeypatch):
    for response in ((403, {}), (200, {"status": "failed", "error": "invalid_submission"})):
        result, _ = invoke(monkeypatch, [(202, {"id": KEY}), response], arguments() + ["--wait"])
        assert result.exit_code == 1


def test_pending_wait_timeout_keeps_identifier_visible(monkeypatch):
    monkeypatch.setattr(cli, "monotonic", Mock(side_effect=[0, 0, 2]))
    result, _ = invoke(
        monkeypatch,
        [(202, {"id": KEY}), (200, {"status": "queued"})],
        arguments() + ["--wait", "--timeout", "1"],
    )
    assert result.exit_code == 1 and KEY in result.output and "Still pending" in result.output
