import json
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest
from sqlalchemy.orm import Session
from test_baseline_ingress import ingress as ingress

from cannery import registration_worker as worker
from cannery.git_intake import SnapshotBinding
from cannery.numerics.baseline_state import BaselineState
from cannery.registry import recipe_registration as recipes
from cannery.registry.registrations import Registration, create_registration
from cannery.settings import Settings


@pytest.fixture
def registration(ingress, monkeypatch, tmp_path):
    _, engine, baseline_key, control, memory, _ = ingress
    with Session(engine) as db:
        binding = SnapshotBinding.model_validate(db.get(BaselineState, baseline_key).snapshot)
    settings = Settings(
        snapshot_root=tmp_path / "snapshots",
        intake_scratch=tmp_path / "scratch",
        baseline_root=tmp_path / "control",
        baseline_volume="test-control",
        artifact_backend="private",
    )
    monkeypatch.setattr(worker, "settings", settings)
    monkeypatch.setattr(worker, "make_engine", lambda _: engine)
    monkeypatch.setattr(worker, "remove_smoke", lambda _: None)
    monkeypatch.setattr(recipes, "make_store", lambda _: memory)
    calls = []

    def intake(*args, **kwargs):
        calls.append("git")
        return json.dumps({"binding": binding.model_dump(mode="json"), "evidence": None}).encode()

    monkeypatch.setattr(worker, "run", intake)

    def create():
        return create_registration(engine, binding.source, "team", "identity", False)

    return engine, baseline_key, control, calls, create, settings


def test_recipe_registration_waits_resumes_without_git_and_freezes_once(registration):
    engine, key, control, calls, create, _ = registration
    identifier = create()
    control["mode"] = "pending"
    worker.register_spec.fn(str(identifier))
    with Session(engine) as db, db.begin():
        row = db.get(Registration, identifier)
        assert row.prepared_snapshot and row.baseline_key == key
        assert row.status == "running" and row.attempts == 1 and row.lease is None
        assert row.cleanup_handle == "oracle:" + key
        row.lease_until = datetime.now(UTC) - timedelta(seconds=1)
        baseline = db.get(BaselineState, key)
        baseline.due_at = baseline.lease_until = datetime.now(UTC) - timedelta(seconds=1)
    control["mode"] = None
    worker.register_spec.fn(str(identifier))
    with Session(engine) as db:
        row = db.get(Registration, identifier)
        assert row.status == "complete" and row.spec_id and row.attempts == 1
    assert calls == ["git"] and control["calls"] == 2
    worker.register_spec.fn(str(create()))
    assert calls == ["git", "git"] and control["calls"] == 2


def test_unconfigured_recipe_registration_fails_without_reference(registration):
    engine, _, control, _, create, settings = registration
    settings.baseline_volume = None
    identifier = create()
    worker.register_spec.fn(str(identifier))
    with Session(engine) as db:
        row = db.get(Registration, identifier)
        assert row.status == "failed" and row.error == "baseline_unconfigured"
    assert control["calls"] == 0


def test_terminal_registration_does_not_cancel_shared_active_oracle(registration, monkeypatch):
    engine, key, _, _, _, settings = registration
    cleanup = MagicMock()
    monkeypatch.setattr("cannery.numerics.baselines.OracleJob", cleanup)
    assert recipes.cleanup_recipe(engine, "oracle:" + key, settings) is False
    cleanup.assert_not_called()


def test_expired_shared_oracle_is_fenced_before_owned_cleanup(registration, monkeypatch):
    from uuid import uuid4

    engine, key, _, _, _, settings = registration
    attempt = uuid4()
    with Session(engine) as db, db.begin():
        row = db.get(BaselineState, key)
        row.status, row.attempt, row.attempts = "generating", attempt, 1
        row.deadline = row.lease_until = datetime.now(UTC) - timedelta(seconds=1)
    cleanup = MagicMock()
    monkeypatch.setattr("cannery.numerics.baselines.OracleJob", cleanup)
    assert recipes.cleanup_recipe(engine, "oracle:" + key, settings)
    cleanup.return_value.cleanup.assert_called_once()
    with Session(engine) as db:
        assert db.get(BaselineState, key).status == "failed"


def test_untrusted_cleanup_handle_rejected(registration):
    engine, _, _, _, _, settings = registration
    with pytest.raises(ValueError, match="cleanup handle"):
        recipes.cleanup_recipe(engine, "oracle:../../foreign", settings)
