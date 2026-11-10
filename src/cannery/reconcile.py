"""Replay durable CPU probes; expire them even if retry callbacks were lost."""

import logging
import time

from redis.exceptions import RedisError
from sqlalchemy.exc import SQLAlchemyError

from cannery.database import make_engine
from cannery.jobs import reconcile_probes
from cannery.registry.dispatch import publish_registration, publish_submission
from cannery.registry.registrations import due_registrations
from cannery.submissions.state import due_intakes
from cannery.worker import probe, settings


def main() -> None:
    engine = make_engine(settings)
    try:
        while True:
            try:
                for key in reconcile_probes(engine):
                    probe.send(key)
                for key in due_registrations(engine):
                    publish_registration(settings.redis_url.get_secret_value(), key)
                for key in due_intakes(engine):
                    publish_submission(settings.redis_url.get_secret_value(), key)
            except (SQLAlchemyError, RedisError):
                logging.warning("Job reconciliation unavailable; retrying in 5 seconds")
            time.sleep(5)
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
