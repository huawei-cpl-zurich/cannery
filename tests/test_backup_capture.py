import copy
import io
import subprocess
import tarfile

import pytest

from cannery.maintenance.archive import verify
from cannery.maintenance.capture import Docker, capture, inventory


def container(service, mounts=(), running=False, project="source"):
    return {
        "Id": service,
        "Image": "sha256:" + "a" * 64,
        "State": {"Running": running, "Status": "running" if running else "exited"},
        "Config": {
            "Labels": {
                "com.docker.compose.project": project,
                "com.docker.compose.service": service,
            },
            "Env": ["POSTGRES_USER=cannery", "POSTGRES_DB=cannery"],
        },
        "Mounts": [
            {"Type": "volume", "Destination": path, "Name": volume, "RW": True}
            for path, volume in mounts
        ],
    }


class Fake(Docker):
    def __init__(self):
        self.containers = [
            container("postgres", running=True),
            container("api", [("/var/lib/cannery/snapshots", "git")]),
            container("artifacts", [("/home/cannery/artifacts", "blobs")]),
            container("builder", [("/var/lib/cannery/builds", "builds")]),
        ]
        self.calls = []
        self.fault = None
        self.inspect_count = 0
        self.after = None

    def run(self, *args, source=None, destination=None, timeout=300):
        self.calls.append(args)
        if args[:2] == ("ps", "-aq"):
            return b"postgres api artifacts builder"
        if args[:2] == ("ps", "-q"):
            return b""
        if "SHOW server_version_num" in args:
            return b"170000"
        if "SELECT version_num FROM alembic_version" in args:
            return b"0013"
        if "pg_dump" in args:
            destination.write(b"PGDMPfixture")
            if self.fault == "database":
                raise ValueError("network disconnected")
            return b""
        raise AssertionError(args)

    def inspect(self, *ids):
        self.inspect_count += 1
        if self.inspect_count == 2 and self.after:
            self.after()
        return copy.deepcopy(self.containers)

    def volume(self, image, name, program, *, destination=None, source=None, writable=False):
        if self.fault == name:
            raise ValueError("volume unavailable")
        data = io.BytesIO()
        with tarfile.open(fileobj=data, mode="w") as archive:
            item = tarfile.TarInfo("fixture")
            item.size = 3
            archive.addfile(item, io.BytesIO(b"yes"))
        destination.write(data.getvalue())
        return b""


def test_backup_includes_durable_journals_and_private_files(tmp_path):
    docker = Fake()
    result = capture(docker, "source", tmp_path / "backup")
    assert {file.role for file in result.files} == {
        "database",
        "snapshots",
        "artifacts",
        "build_control",
    }
    assert verify(tmp_path / "backup") == result
    assert (tmp_path / "backup").stat().st_mode & 0o777 == 0o700
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in (tmp_path / "backup").iterdir())
    assert not any("password" in str(call).lower() for call in docker.calls)


@pytest.mark.parametrize("fault", ["database", "git", "blobs", "builds"])
def test_interrupted_capture_never_publishes_an_incomplete_backup(tmp_path, fault):
    docker = Fake()
    docker.fault = fault
    with pytest.raises(ValueError):
        capture(docker, "source", tmp_path / "backup")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("status", ["running", "paused", "restarting", "removing"])
def test_all_potential_writers_must_be_stopped(tmp_path, status):
    docker = Fake()
    docker.containers[1]["State"] = {"Running": status == "running", "Status": status}
    with pytest.raises(ValueError, match="Stop all"):
        capture(docker, "source", tmp_path / "backup")
    assert not list(tmp_path.iterdir())


def test_changed_project_and_racing_destination_do_not_publish(tmp_path):
    docker = Fake()
    docker.after = lambda: docker.containers[1].update(Image="sha256:" + "b" * 64)
    with pytest.raises(ValueError, match="changed"):
        capture(docker, "source", tmp_path / "backup")
    assert not list(tmp_path.iterdir())
    docker = Fake()
    docker.after = lambda: (tmp_path / "backup").mkdir()
    with pytest.raises(OSError):
        capture(docker, "source", tmp_path / "backup")
    assert list(tmp_path.iterdir()) == [tmp_path / "backup"] and not list(
        (tmp_path / "backup").iterdir()
    )


def test_existing_destination_and_external_volume_writer_are_rejected(tmp_path):
    (tmp_path / "backup").mkdir()
    with pytest.raises(ValueError, match="must not exist"):
        capture(Fake(), "source", tmp_path / "backup")

    class Writer(Fake):
        def run(self, *args, **kwargs):
            return b"api" if args[:2] == ("ps", "-q") else super().run(*args, **kwargs)

    with pytest.raises(ValueError, match="running container"):
        inventory(Writer(), "source")


def test_docker_timeout_is_useful_and_does_not_echo_environment(monkeypatch):
    def fail(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 1)

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(ValueError, match="deadline"):
        Docker().run("inspect", "container", timeout=1)


def test_directory_entries_are_durable_before_backup_becomes_visible(tmp_path, monkeypatch):
    import os
    import stat
    from cannery.maintenance import capture as module

    synced = []
    original_sync = os.fsync
    original_publish = module.publish_directory

    def sync(fd):
        info = os.fstat(fd)
        if stat.S_ISDIR(info.st_mode):
            synced.append(info.st_ino)
        original_sync(fd)

    def publish(source, destination):
        assert source.stat().st_ino in synced
        original_publish(source, destination)

    monkeypatch.setattr(module.os, "fsync", sync)
    monkeypatch.setattr(module, "publish_directory", publish)
    capture(Fake(), "source", tmp_path / "backup")
    assert synced[-1] == tmp_path.stat().st_ino
