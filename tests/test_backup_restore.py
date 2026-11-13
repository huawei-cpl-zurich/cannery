import json

import pytest
from test_backup_capture import Fake
from typer.testing import CliRunner

from cannery.maintenance.archive import verify
from cannery.maintenance.capture import capture
from cannery.maintenance.cli import app
from cannery.maintenance.restore import restore


class Target(Fake):
    def __init__(self):
        super().__init__()
        for c in self.containers:
            c["Config"]["Labels"]["com.docker.compose.project"] = "target"
        self.writes = []
        self.dirty = False
        self.owner = "target"
        self.pg_dirty = False

    def run(self, *args, source=None, **kwargs):
        if args[:2] == ("volume", "inspect"):
            return json.dumps([{"Labels": {"com.docker.compose.project": self.owner}}]).encode()
        if args[-1].startswith("SELECT count(*) FROM pg_class"):
            return b"1" if self.pg_dirty else b"0"
        if "pg_restore" in args:
            self.writes.append(("database", source.read()))
            if self.fault == "database":
                raise ValueError("Database disconnected")
            return b""
        return super().run(*args, source=source, **kwargs)

    def volume(self, image, name, program, *, destination=None, source=None, writable=False):
        if writable:
            self.writes.append((name, source.read()))
            if self.fault == name:
                raise ValueError("Interrupted volume restore")
        elif self.dirty:
            raise ValueError("Restore volume is not empty")
        return b""


@pytest.fixture
def saved(tmp_path):
    capture(Fake(), "source", tmp_path / "backup")
    return tmp_path / "backup"


def test_restore_is_explicit_empty_and_never_starts_workers(saved):
    docker = Target()
    result = restore(docker, saved, "target")
    assert result == verify(saved)
    assert [v[0] for v in docker.writes] == ["git", "blobs", "builds", "database"]
    assert not any("start" in call or "up" in call or "migrate" in call for call in docker.calls)
    assert all("pg_restore" not in call or "--single-transaction" in call for call in docker.calls)


@pytest.mark.parametrize("fault", ["dirty", "pg_dirty", "owner", "source", "changed", "corrupt"])
def test_restore_refuses_unsafe_targets_before_any_writes(saved, fault):
    docker = Target()
    if fault == "dirty":
        docker.dirty = True
    elif fault == "pg_dirty":
        docker.pg_dirty = True
    elif fault == "owner":
        docker.owner = "source"
    elif fault == "changed":
        docker.after = lambda: docker.containers[1].update(Image="sha256:" + "f" * 64)
    elif fault == "corrupt":
        (saved / "database.dump").write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        restore(docker, saved, "source" if fault == "source" else "target")
    assert docker.writes == []


@pytest.mark.parametrize("fault", ["git", "database"])
def test_interrupted_restore_is_failure_and_never_dispatches_work(saved, fault):
    docker = Target()
    docker.fault = fault
    with pytest.raises(ValueError):
        restore(docker, saved, "target")
    assert not any("start" in call for call in docker.calls)


def test_verification_cli_is_read_only_and_errors_are_actionable(saved):
    result = CliRunner().invoke(app, ["verify-backup", str(saved)])
    assert result.exit_code == 0 and json.loads(result.output)["verified"]
    (saved / "database.dump").write_bytes(b"bad")
    result = CliRunner().invoke(app, ["verify-backup", str(saved)])
    assert result.exit_code == 1 and "checksum" in result.output
