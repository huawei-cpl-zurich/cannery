import json
from pathlib import Path
from uuid import uuid4

import pytest
from test_build_sandbox import profile

from cannery.benchmark import device
from cannery.builds.remote import RemotePending


@pytest.fixture
def probe(tmp_path, monkeypatch):
    paths = set()
    calls = []
    now = [100.0]
    lost = [False]

    class Remote:
        def __init__(self, client, journal, target, **kwargs):
            self.journal = journal
            journal.mkdir(parents=True, exist_ok=True)

        def operation(self, name, script, **kwargs):
            paths.add(self.journal.name)
            calls.append(self.journal.name)
            if lost[0]:
                lost[0] = False
                raise RemotePending("reply lost")
            return "remote:test:job:" + self.journal.name

        def call(self, *args):
            return {"content": json.dumps({"hardware_id": "a" * 64, "driver": "b" * 64})}

    monkeypatch.setattr(device, "CplRemote", Remote)
    monkeypatch.setattr(device.time, "time", lambda: now[0])
    config = profile(tmp_path / "worker")
    root = tmp_path / "identity"

    def run(current=None):
        return device.device_cohort(
            current or config, root, Path("/client"), Path("/registry"), "c" * 64
        )

    return run, paths, calls, now, lost, root, config


def test_cache_and_restart_resume_same_discovery_attempt(probe):
    run, paths, calls, now, lost, root, _ = probe
    lost[0] = True
    with pytest.raises(RemotePending):
        run()
    assert (root / "pending.json").exists()
    cohort = run()
    assert len(paths) == 1 and len(calls) == 2 and not (root / "pending.json").exists()
    assert run() == cohort and len(calls) == 2
    now[0] += 61
    assert run() == cohort and len(paths) == 2


def test_profile_change_reconciles_previous_probe_before_new_discovery(probe):
    run, paths, _, _, lost, root, config = probe
    lost[0] = True
    with pytest.raises(RemotePending):
        run()
    changed = config.model_copy(update={"image": "sha256:" + "d" * 64})
    with pytest.raises(RemotePending, match="Previous device probe reconciled"):
        run(changed)
    assert len(paths) == 1
    assert run(changed).image == changed.image and len(paths) == 2
    assert not (root / "pending.json").exists()


def test_completed_probe_cleanup_is_bounded_and_preserves_unknown_attempt(probe):
    run, _, _, now, _, root, _ = probe
    root.mkdir()
    unknown = root / uuid4().hex
    unknown.mkdir()
    for _ in range(5):
        run()
        now[0] += 61
    completed = [p for p in root.iterdir() if p.is_dir() and (p / "done").exists()]
    assert len(completed) == 2 and unknown.exists()
