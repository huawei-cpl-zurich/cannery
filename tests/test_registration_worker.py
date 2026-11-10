from pathlib import Path
from unittest.mock import MagicMock

import pytest

from cannery import registration_worker as worker


def test_duplicate_delivery_does_not_start_attempt(monkeypatch):
    engine = MagicMock()
    monkeypatch.setattr(worker, "make_engine", lambda _: engine)
    monkeypatch.setattr(worker, "pending_cleanup", lambda *a: None)
    monkeypatch.setattr(worker, "claim_registration", lambda *a: None)
    monkeypatch.setattr(worker, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    worker.register_spec.fn("a" * 32)
    engine.dispose.assert_called_once()


def test_terminal_cleanup_is_verified_before_forgetting_handle(monkeypatch, tmp_path):
    engine = MagicMock()
    handle = "cannery-smoke-" + "a" * 32 + "@" + "b" * 32
    monkeypatch.setattr(worker, "make_engine", lambda _: engine)
    monkeypatch.setattr(worker, "pending_cleanup", lambda *a: handle)
    calls = []
    monkeypatch.setattr(worker, "remove_smoke", lambda h: calls.append("verified"))
    monkeypatch.setattr(worker, "discard_scratch", lambda *a: calls.append("scratch"))
    monkeypatch.setattr(worker, "clear_cleanup", lambda *a: calls.append("cleared"))
    worker.register_spec.fn("a" * 32)
    assert calls == ["verified", "scratch", "cleared"]


def test_scratch_path_is_owned_by_validated_handle():
    handle = "cannery-smoke-" + "a" * 32 + "@" + "b" * 32
    assert worker.scratch_for(Path("/tmp/attempts"), handle) == Path("/tmp/attempts") / (
        "attempt-" + "b" * 32
    )


@pytest.mark.parametrize("cleanup_fails", [False, True, "interrupted"])
def test_registration_attempt_freezes_and_completes_atomically(
    monkeypatch, tmp_path, cleanup_fails
):
    import json

    from spec_fixtures import spec_files
    from sqlalchemy import create_engine, select
    from sqlalchemy.orm import Session

    from cannery.database import Base
    from cannery.git_intake import GitSource
    from cannery.git_intake.snapshots import SelectedTree, SnapshotStore
    from cannery.registry.freeze import SmokeEvidence, kernel_key
    from cannery.registry.manifest import validate_spec
    from cannery.registry.models import SpecVersion
    from cannery.registry.registrations import Registration, create_registration
    from cannery.settings import Settings

    engine = create_engine(f"sqlite:///{tmp_path / 'worker.db'}")
    Base.metadata.create_all(engine)
    config = Settings(snapshot_root=tmp_path / "snapshots", intake_scratch=tmp_path / "scratch")
    monkeypatch.setattr(worker, "settings", config)
    monkeypatch.setattr(worker, "make_engine", lambda _: engine)
    monkeypatch.setattr(worker, "remove_smoke", lambda _: None)
    source = GitSource(url="https://example.com/repo.git", commit="a" * 40, path="spec")
    key = create_registration(engine, source, "team", None, False)
    calls = []

    def admitted(argv, **kwargs):
        payload = json.loads(kwargs["data"])
        assert payload["source"] == source.model_dump(mode="json")
        assert kwargs["timeout"] == 150
        assert kwargs["git_limits"] is False
        calls.append(payload["handle"])
        files = spec_files()
        store = SnapshotStore(config.snapshot_root)
        binding = store.commit(
            kernel_key("team", "identity"),
            source,
            SelectedTree(files, dict.fromkeys(files, "100644"), "b" * 40),
        )
        evidence = SmokeEvidence(
            spec_sha256=validate_spec(files, "spec").content_sha256,
            image_id="sha256:" + "c" * 64,
            python="3.12",
            torch="2.6.0+cpu",
            cases=1,
        )
        return json.dumps(
            {
                "binding": binding.model_dump(mode="json"),
                "evidence": evidence.model_dump(mode="json"),
            }
        ).encode()

    first_run = True

    def interrupted(argv, **kwargs):
        nonlocal first_run
        if first_run:
            first_run = False
            calls.append(__import__("json").loads(kwargs["data"])["handle"])
            from cannery.git_intake.process import IntakeError

            raise IntakeError("client killed; creation outcome unknown")
        return admitted(argv, **kwargs)

    monkeypatch.setattr(worker, "run", interrupted if cleanup_fails == "interrupted" else admitted)
    original = worker.discard_scratch
    if cleanup_fails is True:
        monkeypatch.setattr(worker, "discard_scratch", MagicMock(side_effect=OSError("temporary")))
    worker.register_spec.fn(str(key))
    if cleanup_fails:
        from datetime import UTC, datetime

        with Session(engine) as session, session.begin():
            row = session.get(Registration, key)
            assert row.status == "queued" and row.cleanup_handle == calls[0]
            if cleanup_fails is True:
                assert list(config.intake_scratch.iterdir())
            row.due_at = datetime.now(UTC)
        monkeypatch.setattr(worker, "discard_scratch", original)
        worker.register_spec.fn(str(key))
    worker.register_spec.fn(str(key))
    assert len(calls) == (2 if cleanup_fails else 1)
    with Session(engine) as session:
        row = session.get(Registration, key)
        assert row.status == "complete" and row.spec_id is not None
        assert row.cleanup_handle is None
        assert len(session.scalars(select(SpecVersion)).all()) == 1
    assert list(config.intake_scratch.iterdir()) == []
    engine.dispose()


def test_coordinator_preserves_docker_address_space_budget():
    import json
    import resource
    import sys

    from cannery.git_intake.process import run

    probe = "import resource,json; print(json.dumps(resource.getrlimit(resource.RLIMIT_AS)))"
    result = json.loads(run([sys.executable, "-c", probe], git_limits=False))
    assert result == list(resource.getrlimit(resource.RLIMIT_AS))
    assert json.loads(run([sys.executable, "-c", probe])) == [512 * 1024 * 1024] * 2


def test_scratch_cleanup_failure_keeps_recovery_handle(monkeypatch, tmp_path):
    from cannery.registry.sandbox import new_handle

    engine = MagicMock()
    handle = new_handle()
    monkeypatch.setattr(worker, "make_engine", lambda _: engine)
    monkeypatch.setattr(worker, "pending_cleanup", lambda *a: handle)
    monkeypatch.setattr(worker, "remove_smoke", lambda *a: None)
    cleared = MagicMock()
    monkeypatch.setattr(worker, "clear_cleanup", cleared)

    def broken(*args):
        raise OSError("temporary deletion failure")

    monkeypatch.setattr(worker, "discard_scratch", broken)
    worker.register_spec.fn("a" * 32)
    cleared.assert_not_called()
    monkeypatch.setattr(worker, "discard_scratch", lambda *a: None)
    worker.register_spec.fn("a" * 32)
    cleared.assert_called_once_with(engine, __import__("uuid").UUID("a" * 32), handle)
