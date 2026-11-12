import sys
from pathlib import Path

import pytest

from cannery.builds.remote import CplRemote
from cannery.git_intake.process import IntakeError, run


@pytest.mark.parametrize(
    "large,size,success", [(False, 65, False), (True, 65, True), (True, 129, False)]
)
def test_actual_child_file_limit_is_bounded(tmp_path, large, size, success):
    target = tmp_path / "raw"
    script = 'import os,sys;f=open(sys.argv[1],"wb");f.truncate(int(sys.argv[2])*1024*1024)'

    def invoke():
        return run(
            [sys.executable, "-c", script, str(target), str(size)],
            git_limits=False,
            env={"PYTHONPATH": str(Path("src").resolve())},
            large_output=large,
            timeout=10,
        )

    if success:
        invoke()
        assert target.stat().st_size == size * 1024 * 1024
    else:
        with pytest.raises(IntakeError):
            invoke()


def test_git_cannot_request_larger_budget():
    with pytest.raises(ValueError, match="trusted coordinator"):
        run([sys.executable, "-c", "pass"], large_output=True)


def test_transport_only_opts_download_into_larger_budget(tmp_path, monkeypatch):
    client = tmp_path / "skill/scripts/cpl-remote"
    client.parent.mkdir(parents=True)
    (client.parent.parent / "SKILL.md").write_text("test skill")
    remote = CplRemote(client, tmp_path / "journal", "test-a5")
    remote._available = True
    calls = []

    def execute(args, **kwargs):
        calls.append((args, kwargs))
        return b'{"state":"completed"}'

    monkeypatch.setattr("cannery.builds.remote.run", execute)
    remote.transfer("download", "remote/raw", "local/raw", large_output=True)
    assert calls[-1][1]["large_output"] is True
    remote.transfer("download", "remote/raw", "local/raw")
    assert calls[-1][1]["large_output"] is False
    with pytest.raises(ValueError):
        remote.transfer("upload", "local/raw", "remote/raw", large_output=True)
