import asyncio
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient
from redis.exceptions import ConnectionError as RedisConnectionError
from sqlalchemy.exc import OperationalError

from cannery.api import create_app


def dependencies(monkeypatch):
    engine = MagicMock()
    engine.dispose = AsyncMock()
    connection = engine.connect.return_value.__aenter__.return_value
    connection.execute = AsyncMock()
    redis = AsyncMock()
    monkeypatch.setattr("cannery.api.create_async_engine", lambda *a, **kw: engine)
    monkeypatch.setattr("cannery.api.Redis.from_url", lambda *a, **kw: redis)
    return engine, connection, redis


def test_readiness_failure_and_recovery(monkeypatch) -> None:
    engine, connection, redis = dependencies(monkeypatch)
    with TestClient(create_app()) as client:
        assert client.get("/health/ready").status_code == 200
        connection.execute.side_effect = OperationalError("", {}, Exception("secret"))
        redis.ping.side_effect = RedisConnectionError("secret")
        response = client.get("/health/ready")
        assert response.status_code == 503
        assert response.json()["checks"] == {"postgres": False, "redis": False}
        assert "secret" not in response.text
        assert client.get("/health/live").status_code == 200
        connection.execute.side_effect = None
        redis.ping.side_effect = None
        assert client.get("/health/ready").status_code == 200
    engine.dispose.assert_awaited_once()
    redis.aclose.assert_awaited_once()


def test_stalled_connection_is_cancelled_and_recovers(monkeypatch) -> None:
    engine, connection, _ = dependencies(monkeypatch)
    cancelled = []

    async def stalled(*args):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(True)

    connection.execute.side_effect = stalled
    with TestClient(create_app()) as client:
        import time
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor() as pool:
            started = time.monotonic()
            probe = pool.submit(client.get, "/health/ready")
            assert client.get("/health/live").status_code == 200
            assert probe.result(timeout=5).status_code == 503
            assert time.monotonic() - started < 5
        assert cancelled == [True]
        engine.connect.return_value.__aexit__.assert_awaited()
        connection.execute.side_effect = None
        assert client.get("/health/ready").status_code == 200
