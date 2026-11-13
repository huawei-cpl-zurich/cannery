"""Replay durable CPU probes; expire them even if retry callbacks were lost."""

import logging
import threading

from redis.exceptions import RedisError
from sqlalchemy.exc import SQLAlchemyError

from cannery.database import make_engine
from cannery.jobs import reconcile_probes
from cannery.registry.dispatch import publish_registration, publish_submission
from cannery.registry.registrations import due_registrations
from cannery.shutdown import stop_on_term
from cannery.submissions.state import due_intakes
from cannery.worker import probe, settings


def main() -> None:
    engine = make_engine(settings)
    stopped = threading.Event()
    restore_signals = stop_on_term(stopped)
    try:
        while not stopped.is_set():
            try:
                for key in reconcile_probes(engine):
                    if stopped.is_set():
                        break
                    probe.send(key)
                for key in due_registrations(engine):
                    if stopped.is_set():
                        break
                    publish_registration(settings.redis_url.get_secret_value(), key)
                for key in due_intakes(engine):
                    if stopped.is_set():
                        break
                    publish_submission(settings.redis_url.get_secret_value(), key)
            except (SQLAlchemyError, RedisError):
                logging.warning("Job reconciliation unavailable; retrying in 5 seconds")
            stopped.wait(5)
    finally:
        restore_signals()
        engine.dispose()


if __name__ == "__main__":
    main()
