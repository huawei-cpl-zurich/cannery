import json
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from sqlalchemy.orm import Session
from test_candidate_freeze import retained
from test_submission_requests import initial
from test_submission_requests import records as records

from cannery import submission_worker as worker
from cannery.git_intake.process import IntakeError
from cannery.git_intake.snapshots import SnapshotStore
from cannery.settings import Settings
from cannery.submissions.models import Candidate
from cannery.submissions.state import IntakeState


@pytest.mark.parametrize("failure", [None, "scratch", "interrupted", "fenced"])
def test_source_intake_is_fenced_atomic_and_recoverable(records, tmp_path, monkeypatch, failure):
    engine, _ = records
    key = initial(records)
    config = Settings(snapshot_root=tmp_path / "snapshots")
    monkeypatch.setattr(worker, "settings", config)
    monkeypatch.setattr(worker, "make_engine", lambda _: engine)
    calls = []

    def admitted(argv, **kwargs):
        payload = json.loads(kwargs["data"])
        assert argv[-1] == "cannery.submissions.admit" and kwargs["timeout"] == 150
        assert payload["target"] == {"device": "a5", "language": "catlass"}
        calls.append(payload["scratch"])
        if failure == "interrupted" and len(calls) == 1:
            raise IntakeError("injected attempt interruption")
        snapshot = retained(engine, key, SnapshotStore(config.snapshot_root))
        return json.dumps({"snapshot": snapshot.model_dump(mode="json")}).encode()

    monkeypatch.setattr(worker, "run", admitted)
    current = worker.current_intake
    checks = 0

    def fenced(session, claim):
        nonlocal checks
        checks += 1
        if failure == "fenced" and checks == 2:
            return None
        return current(session, claim)

    monkeypatch.setattr(worker, "current_intake", fenced)
    cleanup = worker.discard_scratch
    if failure == "scratch":
        monkeypatch.setattr(
            worker, "discard_scratch", MagicMock(side_effect=OSError("injected cleanup failure"))
        )
    worker.intake_submission.fn(str(key))
    if failure:
        with Session(engine) as session, session.begin():
            state = session.get(IntakeState, key)
            assert state.status == "queued" and state.attempts == 1
            assert session.get(Candidate, key) is None
            if failure == "scratch":
                assert state.scratch_token is not None
            state.due_at = datetime.now(UTC)
        monkeypatch.setattr(worker, "discard_scratch", cleanup)
        worker.intake_submission.fn(str(key))
    worker.intake_submission.fn(str(key))
    assert len(calls) == (2 if failure else 1)
    with Session(engine) as session:
        assert session.get(IntakeState, key).status == "awaiting_build"
        assert session.get(Candidate, key).binding_id == key
        assert session.get(IntakeState, key).scratch_token is None
    assert not list((config.snapshot_root / ".submission-scratch").iterdir())


def test_nonzero_attempt_kills_remaining_process_group(tmp_path):
    import os
    import sys
    import time

    from cannery.git_intake.process import run

    child_pid = tmp_path / "pid"
    script = (
        "import subprocess,sys; p=subprocess.Popen(['sleep','30']); "
        "open(sys.argv[1],'w').write(str(p.pid)); sys.exit(1)"
    )
    with pytest.raises(IntakeError):
        run([sys.executable, "-c", script, str(child_pid)])
    pid = int(child_pid.read_text())
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            status = open(f"/proc/{pid}/stat").read().split()[2]
        except FileNotFoundError:
            break
        if status == "Z":
            break
        time.sleep(0.02)
    else:
        os.kill(pid, 9)
        raise AssertionError("orphan subprocess remained alive")
