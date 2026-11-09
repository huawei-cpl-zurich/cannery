from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from redis import Redis
from redis.exceptions import RedisError
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from cannery.database import make_engine
from cannery.settings import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    config = settings if settings is not None else Settings()
    engine = make_engine(config)
    redis = Redis.from_url(
        config.redis_url.get_secret_value(),
        socket_connect_timeout=1,
        socket_timeout=1,
        retry_on_timeout=False,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        engine.dispose()
        redis.close()

    app = FastAPI(title=config.service_name, lifespan=lifespan)

    @app.get("/health/live")
    def live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready")
    def ready() -> JSONResponse:
        checks = {"postgres": False, "redis": False}
        try:
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            checks["postgres"] = True
        except SQLAlchemyError:
            pass
        try:
            checks["redis"] = bool(redis.ping())
        except RedisError:
            pass
        ok = all(checks.values())
        return JSONResponse(
            {"status": "ok" if ok else "unavailable", "checks": checks},
            status_code=200 if ok else 503,
        )

    return app
