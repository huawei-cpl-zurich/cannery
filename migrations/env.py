from alembic import context

import cannery.jobs  # noqa: F401
import cannery.registry.models  # noqa: F401
from cannery.database import Base, make_engine
from cannery.settings import Settings


def run_migrations() -> None:
    engine = make_engine(Settings())
    try:
        with engine.connect() as connection:
            context.configure(connection=connection, target_metadata=Base.metadata)
            with context.begin_transaction():
                context.run_migrations()
    finally:
        engine.dispose()


run_migrations()
