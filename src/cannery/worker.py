from typing import Any

import dramatiq
from dramatiq.brokers.redis import RedisBroker

from cannery.database import make_engine
from cannery.jobs import complete_probe, fail_probe
from cannery.settings import Settings

settings = Settings()
broker = RedisBroker(  # type: ignore[no-untyped-call]
    url=settings.redis_url.get_secret_value(),
    socket_connect_timeout=2,
    socket_timeout=2,
)
dramatiq.set_broker(broker)


@dramatiq.actor(max_retries=3, min_backoff=1000, max_backoff=5000)
def probe_failed(message: dict[str, Any], retry_data: dict[str, Any]) -> None:
    engine = make_engine(settings)
    try:
        fail_probe(engine, message["args"][0])
    finally:
        engine.dispose()


@dramatiq.actor(
    max_retries=3,
    min_backoff=1000,
    max_backoff=5000,
    on_retry_exhausted="probe_failed",
    time_limit=10000,
)
def probe(job_id: str) -> None:
    engine = make_engine(settings)
    try:
        complete_probe(engine, job_id)
    finally:
        engine.dispose()
