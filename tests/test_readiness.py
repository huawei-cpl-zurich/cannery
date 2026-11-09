from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from redis.exceptions import ConnectionError as RedisConnectionError
from sqlalchemy.exc import OperationalError

from cannery.api import create_app


def test_readiness_failure_and_recovery(monkeypatch) -> None:
    engine = MagicMock()
    redis = MagicMock()
    monkeypatch.setattr("cannery.api.make_engine", lambda _: engine)
    monkeypatch.setattr("cannery.api.Redis.from_url", lambda *a, **kw: redis)
    with TestClient(create_app()) as client:
        assert client.get("/health/ready").status_code == 200
        engine.connect.side_effect = OperationalError("", {}, Exception("secret"))
        redis.ping.side_effect = RedisConnectionError("secret")
        response = client.get("/health/ready")
        assert response.status_code == 503
        assert response.json()["checks"] == {"postgres": False, "redis": False}
        assert "secret" not in response.text
        assert client.get("/health/live").status_code == 200
        engine.connect.side_effect = None
        redis.ping.side_effect = None
        assert client.get("/health/ready").status_code == 200
    engine.dispose.assert_called_once()
    redis.close.assert_called_once()
