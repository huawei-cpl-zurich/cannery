from unittest.mock import AsyncMock

from fastapi.testclient import TestClient
from redis.exceptions import ConnectionError as RedisConnectionError

from cannery.api import create_app


def test_readiness_failure_and_recovery(monkeypatch) -> None:
    postgres = AsyncMock(return_value=True)
    redis = AsyncMock()
    monkeypatch.setattr("cannery.api.postgres_ready", postgres)
    monkeypatch.setattr("cannery.api.Redis.from_url", lambda *a, **kw: redis)
    with TestClient(create_app()) as client:
        assert client.get("/health/ready").status_code == 200
        postgres.return_value = False
        redis.ping.side_effect = RedisConnectionError("secret")
        response = client.get("/health/ready")
        assert response.status_code == 503
        assert response.json()["checks"] == {"postgres": False, "redis": False}
        assert "secret" not in response.text
        assert client.get("/health/live").status_code == 200
        postgres.return_value = True
        redis.ping.side_effect = None
        assert client.get("/health/ready").status_code == 200
    redis.aclose.assert_awaited_once()


def test_stalled_query_closes_socket_before_cancel(monkeypatch) -> None:
    import asyncio
    from unittest.mock import MagicMock

    from cannery.health import postgres_ready

    connection = MagicMock()
    connection.close = AsyncMock()
    cancelled = []

    async def stalled(*args):
        try:
            await asyncio.Event().wait()
        finally:
            connection.pgconn.finish.assert_called_once()
            cancelled.append(True)

    connection.execute = stalled
    monkeypatch.setattr(
        "cannery.health.AsyncConnection.connect", AsyncMock(return_value=connection)
    )
    assert asyncio.run(postgres_ready("postgresql://unused", timeout=0.05)) is False
    assert cancelled == [True]
    connection.close.assert_awaited_once()
    connection.pgconn.finish.reset_mock()
    connection.execute = AsyncMock()
    assert asyncio.run(postgres_ready("postgresql://unused", timeout=0.05)) is True
