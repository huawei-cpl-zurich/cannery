import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from redis.asyncio import Redis
from redis.exceptions import RedisError

from cannery.health import postgres_ready
from cannery.settings import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    config = settings if settings is not None else Settings()
    redis = Redis.from_url(
        config.redis_url.get_secret_value(),
        socket_connect_timeout=1,
        socket_timeout=1,
        retry_on_timeout=False,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        await redis.aclose()

    app = FastAPI(title=config.service_name, lifespan=lifespan)

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready")
    async def ready() -> JSONResponse:
        checks = {"postgres": False, "redis": False}
        checks["postgres"] = await postgres_ready(config.database_url.get_secret_value())
        try:
            async with asyncio.timeout(2):
                checks["redis"] = bool(await redis.ping())
        except (RedisError, TimeoutError):
            pass
        ok = all(checks.values())
        return JSONResponse(
            {"status": "ok" if ok else "unavailable", "checks": checks},
            status_code=200 if ok else 503,
        )

    return app
