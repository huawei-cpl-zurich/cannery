"""Request identity shared by HTTP adapters and application services."""

from contextvars import ContextVar

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy import Engine
from sqlalchemy.exc import SQLAlchemyError
from starlette.concurrency import run_in_threadpool

from cannery.accounts import auth
from cannery.accounts.api import credential
from cannery.settings import Settings

CURRENT: ContextVar[auth.Principal | None] = ContextVar("cannery_principal", default=None)


def identity(settings: Settings) -> str:
    if settings.auth_enabled:
        user = CURRENT.get()
        if user is None:
            raise HTTPException(401, "Authentication required")
        return user.namespace
    if not settings.dev_identity_enabled:
        raise HTTPException(401, "Authentication required")
    return settings.dev_namespace


def role(settings: Settings) -> str:
    if settings.auth_enabled:
        user = CURRENT.get()
        return user.role if user is not None else "anonymous"
    return settings.dev_role if settings.dev_identity_enabled else "anonymous"


def signed_in(settings: Settings) -> bool:
    return CURRENT.get() is not None if settings.auth_enabled else settings.dev_identity_enabled


def template_context(settings: Settings) -> dict[str, object]:
    return {"admin": role(settings) == "admin", "development": settings.dev_identity_enabled}


def public(request: Request) -> bool:
    if request.method not in {"GET", "HEAD"}:
        return False
    path = request.url.path
    return (
        path
        in {
            "/",
            "/global-best",
            "/login",
            "/health/live",
            "/health/ready",
            "/api/v1/releases",
            "/api/v1/releases/kernels",
            "/api/v1/leaderboard",
            "/api/v1/leaderboard/history",
        }
        or path.startswith("/static/")
        or path.startswith("/releases/")
        or (path.startswith("/api/v1/releases/") and len(path.strip("/").split("/")) == 4)
    )


def install_context(app: FastAPI, settings: Settings, engine: Engine) -> None:
    if not settings.auth_enabled:
        return

    @app.middleware("http")
    async def authenticate(request: Request, call_next):  # type: ignore[no-untyped-def]
        token = CURRENT.set(None)
        try:
            if not request.url.path.startswith("/api/v1/auth/"):
                try:
                    raw, kind = credential(request)
                    user = await run_in_threadpool(auth.authenticate, engine, raw, kind)
                    CURRENT.set(user)
                except HTTPException as error:
                    if not public(request):
                        return JSONResponse({"detail": error.detail}, status_code=error.status_code)
                except SQLAlchemyError:
                    return JSONResponse(
                        {"detail": "Authentication temporarily unavailable"}, status_code=503
                    )
            response = await call_next(request)
            if not public(request):
                response.headers["Cache-Control"] = "no-store"
            return response
        finally:
            CURRENT.reset(token)
