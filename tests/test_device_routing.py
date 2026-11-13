import json
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.orm import Session
from test_build_sandbox import profile
from test_submission_requests import records as records

import cannery.builds.state  # noqa: F401
import cannery.numerics.state  # noqa: F401
from cannery.benchmark.contract import Cohort
from cannery.benchmark.state import PerformanceJob, claim_job, due_jobs, ensure_job, postpone_job
from cannery.builds.journals import retained_profile
from cannery.builds.sandbox import WorkerProfile


def test_performance_workers_cannot_claim_or_reclaim_other_device(records):
    engine, _ = records
    for device, key in [("a3", "a" * 64), ("a5", "b" * 64)]:
        ensure_job(engine, key, "reference", {"cohort": {"device": device}})
    assert due_jobs(engine) == ["b" * 64]
    assert due_jobs(engine, device="a3") == ["a" * 64]
    assert claim_job(engine, "a" * 64) is None
    assert claim_job(engine, "b" * 64, device="a3") is None
    claim = claim_job(engine, "a" * 64, device="a3")
    assert claim is not None and postpone_job(engine, claim)
    with Session(engine) as db, db.begin():
        row = db.get(PerformanceJob, claim.key)
        row.due_at = row.lease_until = datetime.now(UTC) - timedelta(seconds=1)
    assert claim_job(engine, claim.key) is None
    resumed = claim_job(engine, claim.key, device="a3")
    assert resumed is not None and resumed.attempt == claim.attempt
    with Session(engine) as db:
        assert db.get(PerformanceJob, "b" * 64).attempts == 0


@pytest.mark.parametrize(
    "field,value", [("device", "a3"), ("target", "other"), ("workspace", "/other/jobs"), ("npu", 7)]
)
def test_cleanup_never_restores_another_workers_route(tmp_path, field, value):
    owner = profile(tmp_path / "worker")
    saved = owner.model_dump()
    saved[field] = value
    if field == "device":
        saved["fingerprint"] = saved["fingerprint"] | {"soc": "Ascend910B4-1"}
    (tmp_path / "context.json").write_text(json.dumps({"profile": saved}))
    assert retained_profile(tmp_path, owner) is None
    assert retained_profile(tmp_path, None) == WorkerProfile.model_validate_json(json.dumps(saved))


def test_cleanup_can_restore_old_toolchain_on_same_device_route(tmp_path):
    owner = profile(tmp_path / "worker")
    saved = WorkerProfile.model_validate(owner.model_dump() | {"image": "sha256:" + "b" * 64})
    assert retained_profile(tmp_path, owner) is None
    (tmp_path / "context.json").write_text(json.dumps({"profile": saved.model_dump(mode="json")}))
    assert retained_profile(tmp_path, owner) == saved


def test_cohort_device_matches_soc_and_keeps_a5_hash_input_unchanged(tmp_path):
    owner = profile(tmp_path / "worker")
    fields = dict(
        hardware_id="physical",
        driver="driver",
        image=owner.image,
        loader_sha256="c" * 64,
        runtime=owner.fingerprint,
    )
    old = Cohort(**fields)
    assert old.model_dump(mode="json")["device"] == "a5"
    with pytest.raises(ValueError, match="device/SoC"):
        Cohort(device="a3", **fields)
    a3 = Cohort(device="a3", **(fields | {"runtime": owner.fingerprint | {"soc": "Ascend910B4-1"}}))
    assert a3.device == "a3"


@pytest.mark.parametrize("stage", ["builds", "numerics", "benchmark"])
def test_foreign_terminal_journals_never_dispatch_cleanup(records, tmp_path, stage):
    from importlib import import_module
    from unittest.mock import MagicMock

    from test_build_state import candidate

    from cannery.benchmark.state import finish_job
    from cannery.builds.state import claim_build, finish_build
    from cannery.numerics.state import NumericalState, claim_numerical, finish_numerical

    engine, _ = records
    key = candidate(records)
    if stage == "builds":
        claim = claim_build(engine, key)
        finish_build(engine, claim, error="deadline_exceeded")
    elif stage == "numerics":
        with Session(engine) as db, db.begin():
            db.add(NumericalState(id=key))
        claim = claim_numerical(engine, key)
        finish_numerical(engine, claim, error="deadline_exceeded")
    else:
        ensure_job(engine, "a" * 64, "reference", {"cohort": {"device": "a5"}})
        claim = claim_job(engine, "a" * 64)
        finish_job(engine, claim, error="deadline_exceeded")
    owner = profile(tmp_path / "worker")
    foreign = WorkerProfile.model_validate(owner.model_dump() | {"target": "other"})
    root = tmp_path / "journals"
    journal = root / claim.attempt.hex
    journal.mkdir(parents=True)
    (journal / "context.json").write_text(json.dumps({"profile": foreign.model_dump(mode="json")}))
    (journal / "runtime.dispatch").write_text("retained")
    factory = MagicMock()
    import_module("cannery." + stage + ".coordinator").cleanup_finished(
        engine, root, factory, owner=owner
    )
    factory.assert_not_called()
    assert (journal / "runtime.dispatch").exists() and not (journal / "cleanup-complete").exists()


def test_build_worker_enables_owned_cleanup_in_once_and_background_modes(tmp_path, monkeypatch):
    import sys
    from unittest.mock import MagicMock

    from cannery.builds import coordinator
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
    monkeypatch.setattr(coordinator, "Settings", lambda: settings)
    monkeypatch.setattr(coordinator, "make_engine", lambda _: MagicMock())
    monkeypatch.setattr(coordinator, "make_store", lambda _: MagicMock())
    monkeypatch.setattr(coordinator, "discover", lambda *args, **kwargs: None)
    monkeypatch.setattr(coordinator, "due_builds", lambda *args, **kwargs: [])
    cleaned = MagicMock()
    monkeypatch.setattr(coordinator, "cleanup_finished", cleaned)
    monkeypatch.setattr(sys, "argv", ["builder", "--once"])
    coordinator.main()
    assert cleaned.call_count == 1 and cleaned.call_args.kwargs["owner"] == config

    def clean_and_stop(*args, **kwargs):
        assert kwargs["owner"] == config
        stopped.set()

    stopped = None

    def record_stop(event):
        nonlocal stopped
        stopped = event
        return lambda: None

    class Cleaner:
        def __init__(self, *, target, daemon):
            self.target = target

        def start(self):
            self.target()

        def is_alive(self):
            return False

    monkeypatch.setattr(coordinator, "stop_on_term", record_stop)
    monkeypatch.setattr(coordinator.threading, "Thread", Cleaner)
    monkeypatch.setattr(coordinator, "cleanup_finished", clean_and_stop)
    monkeypatch.setattr(sys, "argv", ["builder"])
    coordinator.main()
    assert stopped.is_set()
