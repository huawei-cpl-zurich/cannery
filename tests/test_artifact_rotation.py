import hashlib

import pytest
from fastapi.testclient import TestClient

from cannery.artifacts.service import create_app
from cannery.settings import Settings


def test_current_and_previous_roles_remain_separate(tmp_path):
    raw = b"rotation-test"
    path = "/v1/blobs/" + hashlib.sha256(raw).hexdigest()
    config = Settings(
        artifact_root=tmp_path,
        artifact_read_token="r" * 32,
        artifact_write_token="w" * 32,
        artifact_previous_read_token="a" * 32,
        artifact_previous_write_token="b" * 32,
    )
    with TestClient(create_app(config)) as client:
        for value in ["w", "b"]:
            assert (
                client.put(
                    path, content=raw, headers={"Authorization": "Bearer " + value * 32}
                ).status_code
                == 204
            )
            assert (
                client.get(path, headers={"Authorization": "Bearer " + value * 32}).status_code
                == 401
            )
        for value in ["r", "a"]:
            assert (
                client.get(path, headers={"Authorization": "Bearer " + value * 32}).content == raw
            )
            assert (
                client.put(
                    path, content=raw, headers={"Authorization": "Bearer " + value * 32}
                ).status_code
                == 401
            )
    config.artifact_previous_read_token = None
    config.artifact_previous_write_token = None
    with TestClient(create_app(config)) as client:
        assert client.get(path, headers={"Authorization": "Bearer " + "a" * 32}).status_code == 401
        assert (
            client.put(
                path, content=raw, headers={"Authorization": "Bearer " + "b" * 32}
            ).status_code
            == 401
        )
        assert client.get(path, headers={"Authorization": "Bearer " + "r" * 32}).content == raw


@pytest.mark.parametrize(
    "field,value",
    [
        ("artifact_previous_read_token", "w" * 32),
        ("artifact_previous_write_token", "r" * 32),
        ("artifact_previous_read_token", "short"),
    ],
)
def test_invalid_rotations_fail_startup(tmp_path, field, value):
    settings = Settings(
        artifact_root=tmp_path,
        artifact_read_token="r" * 32,
        artifact_write_token="w" * 32,
        **{field: value},
    )
    with pytest.raises(RuntimeError, match="distinct"):
        create_app(settings)
