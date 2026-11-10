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


def test_source_wait_finishes_even_when_build_has_already_finished(monkeypatch):
    result, _ = invoke(
        monkeypatch,
        [
            (202, {"id": KEY}),
            (
                200,
                {
                    "status": "awaiting_validation",
                    "stages": {"intake": "complete", "build": "complete"},
                },
            ),
        ],
        arguments() + ["--wait"],
    )
    assert result.exit_code == 0 and "Source retained" in result.output
    assert "Binary admitted" not in result.output


def test_wait_build_implies_wait_and_observes_binary_admission(monkeypatch):
    result, requests = invoke(
        monkeypatch,
        [
            (202, {"id": KEY}),
            (200, {"status": "building", "stages": {"intake": "complete"}}),
            (200, {"status": "awaiting_validation"}),
        ],
        arguments() + ["--wait-build"],
    )
    assert result.exit_code == 0 and "Binary admitted; awaiting validation" in result.output
    assert len(requests) == 3


def test_wait_build_reports_failure_or_source_only_contract(monkeypatch):
    for status in (
        {"status": "failed", "error": "runtime_failed", "stages": {"intake": "complete"}},
        {
            "status": "awaiting_build",
            "candidate": {"manifest": {"implementation": {"build": {"adapter": "ascendc"}}}},
        },
    ):
        result, _ = invoke(
            monkeypatch, [(202, {"id": KEY}), (200, status)], arguments() + ["--wait-build"]
        )
        assert result.exit_code == 1
