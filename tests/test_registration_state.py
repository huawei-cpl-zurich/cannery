from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine, update
from sqlalchemy.orm import Session

from cannery.database import Base
from cannery.git_intake import GitSource
from cannery.registry.dispatch import publish_registration
from cannery.registry.registrations import (
    Registration,
    claim_registration,
    create_registration,
    current_claim,
    due_registrations,
    fail_registration,
)


@pytest.fixture
def job(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'registration.db'}")
    Base.metadata.create_all(engine)
    key = create_registration(
        engine,
        GitSource(url="https://example.com/repo.git", commit="a" * 40, path="spec"),
        "team",
        None,
        False,
    )
    yield engine, key
    engine.dispose()


def test_duplicate_claims_and_expired_worker_fencing(job):
    engine, key = job
    first = claim_registration(engine, key)
    assert first is not None and claim_registration(engine, key) is None
    with engine.begin() as connection:
        connection.execute(
            update(Registration)
            .where(Registration.id == key)
            .values(lease_until=datetime.now(UTC) - timedelta(seconds=1))
        )
    second = claim_registration(engine, key)
    assert second is not None and second.token != first.token
    assert second.previous_handle is None
    fail_registration(engine, first, "stale_failure", False)
    with Session(engine) as session:
        assert current_claim(session, first) is None
        assert current_claim(session, second) is not None
    fail_registration(engine, second, "invalid_spec", False)
    assert claim_registration(engine, key) is None


def test_retry_backoff_and_terminal_attempt_budget(job):
    engine, key = job
    for attempt in range(3):
        claim = claim_registration(engine, key)
        assert claim is not None
        fail_registration(engine, claim, "source_unavailable", True)
        assert claim_registration(engine, key) is None
        if attempt < 2:
            with engine.begin() as connection:
                connection.execute(
                    update(Registration)
                    .where(Registration.id == key)
                    .values(due_at=datetime.now(UTC) - timedelta(seconds=1))
                )
            assert due_registrations(engine) == [str(key)]
    with Session(engine) as session:
        assert session.get(Registration, key).status == "failed"


def test_expiry_is_durable_and_recoverable_without_redis(job):
    engine, key = job
    with engine.begin() as connection:
        connection.execute(
            update(Registration)
            .where(Registration.id == key)
            .values(deadline=datetime.now(UTC) - timedelta(seconds=1))
        )
    assert due_registrations(engine) == [str(key)]
    assert claim_registration(engine, key) is None
    with Session(engine) as session:
        assert session.get(Registration, key).error == "deadline_exceeded"


def test_dispatch_contains_only_identifier_and_closes_broker(monkeypatch):
    broker = MagicMock()
    monkeypatch.setattr("cannery.registry.dispatch.RedisBroker", lambda **kw: broker)
    publish_registration("redis://test", "job-id")
    message = broker.enqueue.call_args.args[0]
    assert message.args == ("job-id",) and message.kwargs == {}
    assert message.queue_name == "registration"
    broker.close.assert_called_once()


def test_recovery_retains_old_ownership_until_verified_cleanup(job):
    from cannery.registry.registrations import clear_cleanup, prepare_handle

    engine, key = job
    first = claim_registration(engine, key)
    handle = prepare_handle(engine, first)
    with engine.begin() as connection:
        connection.execute(
            update(Registration)
            .where(Registration.id == key)
            .values(lease_until=datetime.now(UTC) - timedelta(seconds=1))
        )
    second = claim_registration(engine, key)
    assert second.previous_handle == handle
    with pytest.raises(ValueError, match="cleanup"):
        prepare_handle(engine, second)
    clear_cleanup(engine, key, handle)
    newer = prepare_handle(engine, second)
    assert newer != handle
    clear_cleanup(engine, key, handle)
    fail_registration(engine, second, "invalid_spec", False)
    with Session(engine) as session:
        assert session.get(Registration, key).cleanup_handle == newer
    assert due_registrations(engine) == [str(key)]


def test_cleanup_backlog_cannot_starve_queued_or_expired_work(job):
    from cannery.registry.sandbox import new_handle

    engine, queued = job
    source = GitSource(url="https://example.com/repo.git", commit="a" * 40, path="spec")
    expired = create_registration(engine, source, "team", None, False)
    assert claim_registration(engine, expired) is not None
    old = datetime.now(UTC) - timedelta(days=1)
    with Session(engine) as session, session.begin():
        for _ in range(120):
            session.add(
                Registration(
                    namespace="team",
                    source=source.model_dump(mode="json"),
                    allow_new_version=False,
                    status="failed",
                    created_at=old,
                    deadline=old,
                    due_at=old,
                    cleanup_handle=new_handle(),
                )
            )
    with engine.begin() as connection:
        connection.execute(
            update(Registration).where(Registration.id == expired).values(lease_until=old)
        )
    observed = set()
    for _ in range(3):
        batch = due_registrations(engine)
        assert str(queued) in batch and str(expired) in batch
        observed.update(set(batch) - {str(queued), str(expired)})
    assert len(observed) == 120
    with Session(engine) as session:
        assert session.get(Registration, queued).status == "queued"
