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


def test_worker_processes_existing_deadlines_when_discovery_is_offline(tmp_path, monkeypatch):
    import sys
    from unittest.mock import MagicMock

    from cannery.benchmark import worker
    from cannery.settings import Settings

    config = profile(tmp_path / "worker")
    path = tmp_path / "profile.json"
    path.write_text(config.model_dump_json())
    settings = Settings(
        artifact_backend="private",
        build_profile=path,
        build_loader_wheel=tmp_path / "loader.whl",
        build_loader_sha256="c" * 64,
        build_control_root=tmp_path / "control",
    )
    engine = MagicMock()
    monkeypatch.setattr(worker, "Settings", lambda: settings)
    monkeypatch.setattr(worker, "make_engine", lambda _: engine)
    monkeypatch.setattr(worker, "make_store", lambda _: MagicMock())
    monkeypatch.setattr(worker, "device_cohort", MagicMock(side_effect=RemotePending("offline")))
    monkeypatch.setattr(worker, "due_jobs", lambda _: ["a" * 64])
    process = MagicMock()
    monkeypatch.setattr(worker, "process", process)
    monkeypatch.setattr(worker, "cleanup_finished", MagicMock())
    monkeypatch.setattr(sys, "argv", ["benchmark-worker", "--once"])
    worker.main()
    assert process.call_args.args[:2] == (engine, "a" * 64)
    engine.dispose.assert_called_once()


def test_failed_probe_retention_is_bounded_during_persistent_outage(probe, monkeypatch):
    from cannery.builds.remote import RemoteFailure

    run, paths, _, _, _, root, _ = probe
    root.mkdir()
    unknown = root / uuid4().hex
    unknown.mkdir()
    original = device.CplRemote.operation

    def failed(self, *args, **kwargs):
        original(self, *args, **kwargs)
        raise RemoteFailure(1)

    monkeypatch.setattr(device.CplRemote, "operation", failed)
    for _ in range(5):
        with pytest.raises(RemotePending, match="Device identity unavailable"):
            run()
    completed = [p for p in root.iterdir() if p.is_dir() and (p / "done").exists()]
    assert len(paths) == 5 and len(completed) == 2 and unknown.exists()
    assert not (root / "pending.json").exists()
