import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from redis.asyncio import Redis
from redis.exceptions import RedisError

from cannery.accounts.api import install_auth
from cannery.accounts.context import install_context
from cannery.artifacts.factory import make_store
from cannery.database import make_engine
from cannery.health import postgres_ready
from cannery.registry.api import install_registry
from cannery.releases.api import install_publications
from cannery.releases.discovery_api import install_discovery
from cannery.releases.download_api import install_downloads
from cannery.settings import Settings
from cannery.submissions.api import install_submissions
from cannery.web import install_web


def create_app(settings: Settings | None = None) -> FastAPI:
    config = settings if settings is not None else Settings()
    if config.release_enabled and config.artifact_backend != "private":
        raise ValueError("Releases require private artifact storage")
    if config.auth_enabled and config.dev_identity_enabled:
        raise ValueError("Account authentication and development identity cannot be combined")
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
        await redis.aclose()
        engine.dispose()

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

    app.state.artifacts = make_store(config)
    install_auth(app, config, engine)
    install_context(app, config, engine)
    install_registry(app, config, engine)
    install_submissions(app, config, engine)
    install_publications(app, config, engine)
    install_discovery(app, config, engine)
    install_downloads(app, config, engine)
    install_web(app, config, engine)
    return app
