import httpx
from typer.testing import CliRunner

from cannery.cli import app

Client = httpx.Client


def invoke(monkeypatch, replies, *options):
    def handle(request):
        status, body = replies.pop(0)
        return httpx.Response(status, json=body)

    client = Client(transport=httpx.MockTransport(handle))
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client)
    monkeypatch.setattr("time.sleep", lambda *a: None)
    return CliRunner().invoke(
        app,
        [
            "register",
            "--repo",
            "https://example.com/repo.git",
            "--commit",
            "a" * 40,
            "--path",
            "spec/",
            *options,
        ],
    )


def test_cli_wait_recovers_transient_status_failure(monkeypatch):
    replies = [
        (202, {"id": "key"}),
        (503, {}),
        (200, {"status": "queued"}),
        (
            200,
            {
                "status": "complete",
                "spec": {"kernel": {"namespace": "team", "name": "identity"}, "version": 1},
            },
        ),
    ]
    result = invoke(monkeypatch, replies, "--wait")
    assert result.exit_code == 0 and "team/identity@spec-v1" in result.output
    assert not replies


def test_cli_denial_and_failed_registration_exit_nonzero(monkeypatch):
    result = invoke(monkeypatch, [(403, {})])
    assert result.exit_code == 1
    result = invoke(monkeypatch, [(202, {"id": "key"}), (200, {"status": "failed"})], "--wait")
    assert result.exit_code == 1 and "Registration failed" in result.output


def test_cli_does_not_retry_status_access_denial(monkeypatch):
    replies = [(202, {"id": "key"}), (404, {})]
    assert invoke(monkeypatch, replies, "--wait").exit_code == 1
    assert not replies
