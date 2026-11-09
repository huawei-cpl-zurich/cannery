from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase

from cannery.settings import Settings


class Base(DeclarativeBase):
    pass


def make_engine(settings: Settings) -> Engine:
    return create_engine(
        settings.database_url.get_secret_value(),
        pool_pre_ping=True,
        pool_timeout=2,
        connect_args={"connect_timeout": 2, "options": "-c statement_timeout=2000"},
    )
