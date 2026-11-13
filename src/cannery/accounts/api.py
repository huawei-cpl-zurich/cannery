"""Authentication endpoints; namespace resource authorization is installed separately."""

import hashlib
import hmac
from typing import Any
from uuid import UUID

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import Field
from redis import Redis
from redis.exceptions import RedisError
from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session

from cannery.accounts import auth
from cannery.accounts.models import Account, Credential
from cannery.registry.manifest import Name, StrictModel
from cannery.settings import Settings

COOKIE = "cannery_session"
SAFE = {"GET", "HEAD", "OPTIONS"}


class Login(StrictModel):
    username: Name
    password: str = Field(min_length=1, max_length=1024)
    namespace: Name | None = None


class Namespace(StrictModel):
    namespace: Name


class Token(Namespace):
    label: str = Field(default="", max_length=64)


class Password(StrictModel):
    current_password: str = Field(min_length=1, max_length=1024)
    new_password: str = Field(min_length=12, max_length=1024)


def credential(request: Request) -> tuple[str, str]:
    authorization = request.headers.get("authorization")
    if authorization is not None:
        prefix, separator, value = authorization.partition(" ")
        if prefix.lower() != "bearer" or not separator or not value or len(value) > 128:
            raise HTTPException(401, "Invalid credentials")
        return value, "token"
    raw = request.cookies.get(COOKIE, "")
    if not raw or len(raw) > 128:
        raise HTTPException(401, "Authentication required")
    if request.method not in SAFE and not hmac.compare_digest(
        request.headers.get("x-csrf-token", "").encode(), auth.csrf(raw).encode()
    ):
        raise HTTPException(403, "CSRF verification failed")
    return raw, "session"


def principal(request: Request, engine: Engine) -> auth.Principal:
    return auth.authenticate(engine, *credential(request))


def origin(request: Request, settings: Settings) -> None:
    # Login CSRF cannot rely on an authenticated session. Require same-origin
    # custom-header requests; CORS is deliberately not enabled.
    if request.headers.get("x-cannery-login") != "1":
        raise HTTPException(403, "Login header required")
    supplied = request.headers.get("origin")
    expected = settings.auth_origin
    if supplied is not None and supplied != expected:
        raise HTTPException(403, "Origin access denied")


def throttle(request: Request, username: str, settings: Settings) -> None:
    client = request.client.host if request.client else "unknown"
    redis = Redis.from_url(
        settings.redis_url.get_secret_value(),
        socket_timeout=1,
        socket_connect_timeout=1,
        retry_on_timeout=False,
    )
    try:
        for scope in ("account:" + username, "client:" + client):
            key = "cannery:login:" + hashlib.sha256(scope.encode()).hexdigest()
            # Atomic expiry, including recovery if a previous client disconnected.
            count = redis.eval(
                "local n=redis.call('INCR',KEYS[1]); "
                "if redis.call('TTL',KEYS[1])<0 then "
                "redis.call('EXPIRE',KEYS[1],300) end; return n",
                1,
                key,
            )
            if int(count) > (10 if scope.startswith("account:") else 50):
                raise HTTPException(429, "Too many login attempts", headers={"Retry-After": "300"})
    except RedisError:
        raise HTTPException(503, "Login temporarily unavailable") from None
    finally:
        redis.close()


def install_auth(app: FastAPI, settings: Settings, engine: Engine) -> None:
    if not settings.auth_enabled:
        return

    @app.post("/api/v1/auth/login")
    def login(body: Login, request: Request, response: Response) -> dict[str, str]:
        origin(request, settings)
        throttle(request, body.username, settings)
        raw = auth.login(engine, body.username, body.password, body.namespace)
        response.set_cookie(
            COOKIE,
            raw,
            max_age=43200,
            httponly=True,
            secure=settings.auth_cookie_secure,
            samesite="strict",
        )
        response.headers["Cache-Control"] = "no-store"
        return {"csrf_token": auth.csrf(raw)}

    @app.get("/api/v1/auth/me")
    def me(request: Request, response: Response) -> dict[str, Any]:
        raw, kind = credential(request)
        user = auth.authenticate(engine, raw, kind)
        with Session(engine) as db:
            memberships = auth.namespaces(db, user.account_id)
        response.headers["Cache-Control"] = "no-store"
        return {
            "username": user.username,
            "role": user.role,
            "namespace": user.namespace,
            "namespaces": memberships,
            "csrf_token": auth.csrf(raw) if kind == "session" else None,
        }

    @app.post("/api/v1/auth/logout", status_code=204)
    def logout(request: Request, response: Response) -> None:
        with Session(engine) as db, db.begin():
            user = auth.resolve(db, *credential(request))
            db.execute(delete(Credential).where(Credential.id == user.credential_id))
        response.delete_cookie(COOKIE)

    @app.post("/api/v1/auth/password", status_code=204)
    def password(body: Password, request: Request, response: Response) -> None:
        auth.reset_own_password(
            engine, *credential(request), body.current_password, body.new_password
        )
        response.delete_cookie(COOKIE)

    @app.post("/api/v1/auth/namespace", status_code=204)
    def namespace(body: Namespace, request: Request) -> None:
        with Session(engine) as db, db.begin():
            user = auth.resolve(db, *credential(request))
            if user.kind != "session":
                raise HTTPException(403, "Tokens are bound to one namespace")
            account = db.get(Account, user.account_id)
            assert account is not None
            if not auth.permitted(db, account, body.namespace):
                raise HTTPException(403, "Namespace access denied")
            stored = db.get(Credential, user.credential_id)
            assert stored is not None
            stored.namespace = body.namespace

    @app.post("/api/v1/auth/tokens", status_code=201)
    def token(body: Token, request: Request, response: Response) -> dict[str, str]:
        with Session(engine) as db, db.begin():
            user = auth.resolve(db, *credential(request))
            if user.kind != "session":
                raise HTTPException(403, "A browser session is required to issue tokens")
            account = db.get(Account, user.account_id)
            assert account is not None
            raw = auth.issue(db, account, "token", body.namespace, body.label)
        response.headers["Cache-Control"] = "no-store"
        return {"token": raw}

    @app.get("/api/v1/auth/tokens")
    def tokens(request: Request, response: Response) -> list[dict[str, str]]:
        user = principal(request, engine)
        response.headers["Cache-Control"] = "no-store"
        with Session(engine) as db:
            return [
                {
                    "id": str(row.id),
                    "label": row.label,
                    "namespace": row.namespace,
                    "expires_at": row.expires_at.isoformat(),
                }
                for row in db.scalars(
                    select(Credential).where(
                        Credential.account_id == user.account_id, Credential.kind == "token"
                    )
                )
            ]

    @app.delete("/api/v1/auth/tokens/{identifier}", status_code=204)
    def revoke(identifier: UUID, request: Request) -> None:
        with Session(engine) as db, db.begin():
            user = auth.resolve(db, *credential(request))
            db.execute(
                delete(Credential).where(
                    Credential.id == identifier,
                    Credential.account_id == user.account_id,
                    Credential.kind == "token",
                )
            )
