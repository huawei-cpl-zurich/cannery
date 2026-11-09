"""Replay durable CPU probes; expire them even if retry callbacks were lost."""

import logging
import time

from redis.exceptions import RedisError
from sqlalchemy.exc import SQLAlchemyError

from cannery.database import make_engine
from cannery.jobs import reconcile_probes
from cannery.worker import probe, settings


def main() -> None:
    engine = make_engine(settings)
    try:
        while True:
            try:
                for key in reconcile_probes(engine):
                    probe.send(key)
            except (SQLAlchemyError, RedisError):
                logging.warning("Probe reconciliation unavailable; retrying in 5 seconds")
            time.sleep(5)
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
