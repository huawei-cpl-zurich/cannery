import json

import pytest
from test_backup_capture import Fake

from cannery.maintenance.drain import STAGES, require_drained


class Deployment(Fake):
    def __init__(self):
        super().__init__()
        self.active = dict.fromkeys(STAGES, 0)

    def run(self, *args, **kwargs):
        if "json_object_agg(stage,active)" in args[-1]:
            return json.dumps(self.active).encode()
        return super().run(*args, **kwargs)


@pytest.mark.parametrize("stage", STAGES)
def test_queued_or_active_work_requires_old_deployment_recovery(stage):
    docker = Deployment()
    docker.active[stage] = 1
    with pytest.raises(ValueError, match="OLD deployment"):
        require_drained(docker, "source")


def test_drain_check_requires_stopped_writers_and_available_database():
    docker = Deployment()
    assert require_drained(docker, "source") == dict.fromkeys(STAGES, 0)
    docker.containers[1]["State"] = {"Running": True, "Status": "running"}
    with pytest.raises(ValueError):
        require_drained(docker, "source")

    class Offline(Deployment):
        def run(self, *args, **kwargs):
            if "json_object_agg(stage,active)" in args[-1]:
                raise ValueError("Database disconnected")
            return super().run(*args, **kwargs)

    with pytest.raises(ValueError, match="disconnected"):
        require_drained(Offline(), "source")
    docker = Deployment()
    docker.active = {"unavailable": 0}
    with pytest.raises(ValueError):
        require_drained(docker, "source")
