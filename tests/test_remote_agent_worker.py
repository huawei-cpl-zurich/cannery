import json
from pathlib import Path, PurePosixPath
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from test_build_executor import setup_executor as setup_executor
from test_build_remote import FakeCpl

from cannery.builds.executor import ExecutionFailure
from cannery.builds.remote import RemoteFailure, RemotePending


class Agent(FakeCpl):
    def __init__(self, root, replies):
        super().__init__(root, replies)
        self._backend = "remote-agent"
        self._workspace = PurePosixPath("/")

    def call(self, *args, **kwargs):
        self.calls.append(args)
        if args[0] in {"upload", "run"}:
            Path(args[args.index("--receipt-file") + 1]).write_text(
                json.dumps(
                    {
                        "target": self.target,
                        "handle": "remote:test-a5:"
                        + ("job" if args[0] == "run" else "transfer")
                        + ":fixture",
                    }
                )
            )
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def test_upload_disconnect_and_restart_observe_same_transfer(tmp_path):
    first = Agent(tmp_path, [RemotePending("lost acknowledgement")])
    with pytest.raises(RemotePending):
        first.transfer("upload", "bundle", "/remote/bundle")
    resumed = Agent(tmp_path, [{"state": "running"}, {"state": "completed"}])
    with pytest.raises(RemotePending):
        resumed.transfer("upload", "bundle", "/remote/bundle")
    resumed.transfer("upload", "bundle", "/remote/bundle")
    assert all(call[0] == "observe" for call in resumed.calls)


def test_download_materializes_retained_bytes_without_resubmission(tmp_path):
    remote = Agent(
        tmp_path,
        [
            {"target": "test-a5", "handle": "remote:test-a5:transfer:fixture", "state": "queued"},
            {"state": "running"},
            {"state": "completed"},
            RemotePending("fetch disconnected"),
            {"state": "completed"},
            {"state": "completed"},
        ],
    )
    for _ in range(2):
        with pytest.raises(RemotePending):
            remote.transfer("download", "/remote/package", "local-package")
    remote.transfer("download", "/remote/package", "local-package")
    assert [call[0] for call in remote.calls].count("download") == 1
    assert [call[0] for call in remote.calls].count("fetch") == 2


def test_terminal_transfer_retries_are_bounded_and_receipts_retained(tmp_path):
    remote = Agent(
        tmp_path, sum(([{"state": "queued"}, {"state": "failed"}] for _ in range(3)), [])
    )
    for _ in range(3):
        with pytest.raises(RemotePending):
            remote.transfer("upload", "source", "/destination")
    with pytest.raises(RemoteFailure):
        remote.transfer("upload", "source", "/destination")
    assert len(list(tmp_path.glob("transfer-*.dispatch"))) == 3
    assert [call[0] for call in remote.calls].count("upload") == 3


def test_missing_transfer_receipt_blocks_ambiguous_dispatch(tmp_path):
    remote = Agent(tmp_path, [RemotePending()])
    with pytest.raises(RemotePending):
        remote.transfer("download", "/remote", "local")
    for _ in range(2):
        with pytest.raises(RemotePending, match="ambiguous"):
            remote.transfer("download", "/remote", "local")
    assert len(remote.calls) == 1


@pytest.mark.parametrize("corrupt", [False, True])
def test_async_upload_bundle_is_immutable_and_checked_before_reuse(setup_executor, corrupt):
    executor, _, remote, _ = setup_executor
    remote.transfer = MagicMock(side_effect=[RemotePending(), None])
    attempt = uuid4()
    with pytest.raises(RemotePending):
        executor.upload(attempt, "source", {"tree/input.cpp": b"pinned source"})
    bundle = remote.journal / "source.tar.gz"
    previous = bundle.read_bytes(), bundle.stat().st_ino
    if corrupt:
        bundle.write_bytes(b"corrupt")
        with pytest.raises(ExecutionFailure, match="transfer_bundle_digest_mismatch"):
            executor.upload(attempt, "source", {"tree/input.cpp": b"pinned source"})
        assert remote.transfer.call_count == 1
    else:
        executor.upload(attempt, "source", {"tree/input.cpp": b"pinned source"})
        assert (bundle.read_bytes(), bundle.stat().st_ino) == previous
        assert "unpack-source" in remote.started


@pytest.mark.parametrize("direction", ["upload", "download"])
def test_remote_paths_are_relative_to_registered_workspace(tmp_path, direction):
    remote = Agent(tmp_path, [RemotePending()])
    remote._workspace = PurePosixPath("/registered/workspace")
    source, destination = (
        ("local", "/registered/workspace/attempt/input")
        if direction == "upload"
        else ("/registered/workspace/attempt/output", "local")
    )
    with pytest.raises(RemotePending):
        remote.transfer(direction, source, destination)
    assert remote.calls[0][3 if direction == "upload" else 2] == "attempt/" + (
        "input" if direction == "upload" else "output"
    )


@pytest.mark.parametrize(
    "path",
    ["/outside/input", "/registered/workspace/../escape", "/registered/workspace", "relative"],
)
def test_remote_paths_cannot_escape_workspace(tmp_path, path):
    remote = Agent(tmp_path, [])
    remote._workspace = PurePosixPath("/registered/workspace")
    with pytest.raises(ValueError):
        remote.transfer("upload", "local", path)
    assert not remote.calls and not list(tmp_path.glob("*.dispatch"))


def test_operation_uses_relative_cwd_without_runtime_override(tmp_path):
    remote = Agent(tmp_path, [{"state": "queued"}, {"state": "completed", "exit": 0}])
    remote._workspace = PurePosixPath("/registered/workspace")
    assert (
        remote.operation(
            "compile", b"print(1)", workspace="/registered/workspace/jobs", runtime=None
        )
        == "remote:test-a5:job:fixture"
    )
    call = remote.calls[0]
    assert call[call.index("--cwd") + 1] == "jobs" and "--runtime" not in call


@pytest.mark.parametrize(
    "backend,workspace,valid",
    [
        ("remote-agent", "/registered/workspace", True),
        ("ssh", "/registered/workspace", False),
        ("remote-agent", "relative", False),
        ("remote-agent", "/registered/../escape", False),
    ],
)
def test_agent_preflight_requires_matching_registered_route(tmp_path, backend, workspace, valid):
    from cannery.builds.remote import CplRemote

    remote = Agent(
        tmp_path / "journal",
        [
            {
                "state": "available",
                "backend": "remote-agent",
                "capabilities": dict.fromkeys(
                    ["run", "observe", "upload", "download", "cancel"], True
                ),
            },
            {"state": "completed"},
        ],
    )
    remote._available = False
    remote.registry = tmp_path / "registry.json"
    remote.registry.write_text(
        json.dumps({"targets": {remote.target: {"backend": backend, "workspace": workspace}}})
    )
    if valid:
        CplRemote.preflight(remote)
        assert remote._workspace == PurePosixPath(workspace) and remote._available
    else:
        with pytest.raises(ValueError):
            CplRemote.preflight(remote)
        assert not remote._available
