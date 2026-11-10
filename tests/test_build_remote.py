import json
from pathlib import Path

import pytest

from cannery.builds.remote import CplRemote, RemoteFailure, RemotePending


class FakeCpl(CplRemote):
    def __init__(self, root, replies):
        super().__init__(Path("/unused/scripts/cpl-remote"), root, "test-a5")
        self.replies, self.calls = replies, []

    def call(self, *args):
        self.calls.append(args)
        if args[0] == "run":
            receipt = Path(args[args.index("--receipt-file") + 1])
            receipt.write_text(
                json.dumps({"target": "test-a5", "handle": "remote:test-a5:job:123"})
            )
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def test_lost_launch_response_recovers_same_handle_after_restart(tmp_path):
    first = FakeCpl(tmp_path, [RemotePending()])
    with pytest.raises(RemotePending):
        first.operation("compile", b"print(1)", workspace="/work", runtime="cann")
    assert first.calls[0][0] == "run"
    restarted = FakeCpl(tmp_path, [{"state": "running"}, {"state": "completed", "exit": 0}])
    with pytest.raises(RemotePending):
        restarted.operation(
            "compile", b"changed code must not dispatch", workspace="/work", runtime="cann"
        )
    handle = restarted.operation("compile", b"print(1)", workspace="/work", runtime="cann")
    assert handle == "remote:test-a5:job:123"
    assert all(call[0] == "observe" for call in restarted.calls)
    assert (tmp_path / "compile.py").read_bytes() == b"print(1)"


def test_ambiguous_dispatch_without_receipt_never_resubmits(tmp_path):
    remote = FakeCpl(tmp_path, [])
    (tmp_path / "compile.dispatch").write_text("dispatching")
    for _ in range(3):
        with pytest.raises(RemotePending, match="ambiguous"):
            remote.operation("compile", b"print(1)", workspace="/work", runtime="cann")
    assert not remote.calls


def test_terminal_failure_and_unavailable_observation_preserve_receipt(tmp_path):
    remote = FakeCpl(
        tmp_path,
        [
            {"state": "running"},
            {"state": "observation-unavailable"},
            {"state": "failed", "exit": 75},
        ],
    )
    with pytest.raises(RemotePending):
        remote.operation("runtime", b"print(1)", workspace="/work", runtime="cann", npu=1)
    assert "--npu" in remote.calls[0]
    with pytest.raises(RemoteFailure) as failure:
        remote.operation("runtime", b"print(1)", workspace="/work", runtime="cann", npu=1)
    assert failure.value.exit_code == 75
    assert len([call for call in remote.calls if call[0] == "run"]) == 1


def test_incomplete_transfer_blocks_dependent_work(tmp_path):
    remote = FakeCpl(tmp_path, [{"state": "failed"}, {"state": "completed"}])
    with pytest.raises(RemotePending):
        remote.transfer("upload", "local/", "/remote/")
    remote.transfer("upload", "local/", "/remote/")
    assert remote.calls[0] == remote.calls[1]


def test_wrong_target_receipt_is_never_observed(tmp_path):
    remote = FakeCpl(tmp_path, [])
    (tmp_path / "compile.json").write_text(
        json.dumps({"target": "other", "handle": "remote:other:job:123"})
    )
    with pytest.raises(RemotePending):
        remote.operation("compile", b"print(1)", workspace="/work", runtime="cann")
    assert not remote.calls


def test_real_process_preserves_nonzero_terminal_json_and_drops_service_secrets(
    tmp_path, monkeypatch
):
    root = tmp_path / "skill"
    (root / "scripts").mkdir(parents=True)
    (root / "SKILL.md").write_text("Read before use")
    client = root / "scripts" / "cpl-remote"
    client.write_text("""#!/usr/bin/python3
import os,json,sys
assert "CANNERY_ARTIFACT_WRITE_TOKEN" not in os.environ
print(json.dumps({"state":"failed","exit":75}))
sys.exit(75)
""")
    client.chmod(0o700)
    monkeypatch.setenv("CANNERY_ARTIFACT_WRITE_TOKEN", "secret")
    remote = CplRemote(client, tmp_path / "journal", "test-a5")
    assert remote.call("observe", "remote:test-a5:job:123") == {"state": "failed", "exit": 75}
