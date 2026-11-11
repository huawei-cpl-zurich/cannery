import sys
import time
from pathlib import Path

import pytest

from cannery.git_intake import GitSource
from cannery.git_intake.process import IntakeError, git, run
from cannery.git_intake.snapshots import SelectedTree, SnapshotStore


def test_outer_deadline_kills_nested_git_process_group(tmp_path):
    marker = tmp_path / "child.pid"
    child = (
        "import os,time,pathlib;pathlib.Path("
        + repr(str(marker))
        + ").write_text(str(os.getpid()));time.sleep(30)"
    )
    program = (
        "from cannery.git_intake.process import attempt_budget,run;import sys;"
        "attempt_budget(10);run([sys.executable,'-c'," + repr(child) + "],timeout=20)"
    )
    with pytest.raises(IntakeError):
        run([sys.executable, "-c", program], timeout=1)
    assert marker.exists()
    pid = int(marker.read_text())
    status = Path(f"/proc/{pid}/stat")
    deadline = time.monotonic() + 2
    while True:
        try:
            state = status.read_text().split()[2]
        except (FileNotFoundError, ProcessLookupError):
            break  # The child has already been reaped.
        if state in {"Z", "X"}:
            break  # Zombie/dead tasks cannot execute further work.
        assert time.monotonic() < deadline, f"Child survived group cancellation: {state}"
        time.sleep(0.01)


def test_maximum_file_count_and_stale_owned_locks_recover(tmp_path):
    store = SnapshotStore(tmp_path / "store", scratch=tmp_path)
    files = {f"spec/file{i}": f"{i}".encode() for i in range(256)}
    source = GitSource(url="https://example.com/repo.git", commit="a" * 40, path="spec")
    selected = SelectedTree(files, dict.fromkeys(files, "100644"), "b" * 40)
    first = store.commit("c" * 64, source, selected)
    repository = store.root / ("c" * 64 + ".git")
    (repository / "refs/heads/accepted.lock").write_text("interrupted")
    (repository / "refs/cannery/snapshots" / ("d" * 64 + ".lock")).write_text("interrupted")
    again = store.commit("c" * 64, source, selected)
    assert first == again and store.read(again).files == files
    assert not (repository / "refs/heads/accepted.lock").exists()
    assert git(repository, "cat-file", "-t", first.internal_snapshot_sha).strip() == b"commit"
