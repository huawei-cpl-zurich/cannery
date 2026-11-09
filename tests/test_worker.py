from datetime import UTC, datetime, timedelta

import dramatiq
import pytest
from dramatiq.brokers.stub import StubBroker
from sqlalchemy import create_engine, select, update

from cannery import worker
from cannery.database import Base
from cannery.jobs import ProbeJob, create_probe, reconcile_probes


@pytest.mark.parametrize("failures,expected", [(2, "complete"), (10, "failed"), (20, "queued")])
def test_broker_retry_and_exhaustion(tmp_path, monkeypatch, failures, expected) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'worker.db'}")
    Base.metadata.create_all(engine)
    monkeypatch.setattr(worker, "make_engine", lambda _: engine)
    original = worker.complete_probe
    calls = []

    def flaky(engine, key):
        calls.append(key)
        if len(calls) <= failures:
            raise RuntimeError("injected database failure")
        original(engine, key)

    monkeypatch.setattr(worker, "complete_probe", flaky)
    if failures == 20:

        def failed_callback(*args):
            raise RuntimeError("injected persistent database outage")

        monkeypatch.setattr(worker, "fail_probe", failed_callback)
    broker = StubBroker()
    options = {"max_retries": 3, "min_backoff": 0, "max_backoff": 0}
    dramatiq.actor(worker.probe_failed.fn, broker=broker, **options)
    actor = dramatiq.actor(
        worker.probe.fn, broker=broker, on_retry_exhausted="probe_failed", **options
    )
    key = create_probe(engine)
    consumer = dramatiq.Worker(broker, worker_timeout=20, worker_threads=2)
    consumer.start()
    try:
        actor.send(str(key))
        broker.join("default", timeout=5000, fail_fast=False)
        consumer.join()
        with engine.connect() as connection:
            row = connection.execute(select(ProbeJob).where(ProbeJob.id == key)).one()
            assert row.status == expected
            assert row.effects == (1 if expected == "complete" else 0)
        assert len(calls) == (3 if expected == "complete" else 4)
        if expected == "queued":
            # Once the database recovers, even a lost exhaustion callback
            # cannot leave an expired durable probe queued forever.
            with engine.begin() as connection:
                connection.execute(
                    update(ProbeJob)
                    .where(ProbeJob.id == key)
                    .values(created_at=datetime.now(UTC) - timedelta(seconds=61))
                )
            assert reconcile_probes(engine) == []
            with engine.connect() as connection:
                assert (
                    connection.scalar(select(ProbeJob.status).where(ProbeJob.id == key)) == "failed"
                )
    finally:
        consumer.stop(timeout=5000)
        broker.close()
        engine.dispose()


def test_reconciliation_recovers_lost_messages_and_expires_jobs(tmp_path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'recovery.db'}")
    Base.metadata.create_all(engine)
    recent = create_probe(engine)
    expired = create_probe(engine)
    with engine.begin() as connection:
        connection.execute(
            update(ProbeJob)
            .where(ProbeJob.id == expired)
            .values(created_at=datetime.now(UTC) - timedelta(seconds=61))
        )
    assert reconcile_probes(engine) == [str(recent)]
    with engine.connect() as connection:
        assert connection.scalar(select(ProbeJob.status).where(ProbeJob.id == expired)) == "failed"
    engine.dispose()
