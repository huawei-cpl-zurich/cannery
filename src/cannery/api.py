from fastapi import FastAPI

from cannery.settings import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    config = settings if settings is not None else Settings()
    app = FastAPI(title=config.service_name)

    @app.get("/health/live")
    def live() -> dict[str, str]:
        return {"status": "ok"}

    return app
