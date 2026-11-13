import hashlib
import io
import tarfile

import pytest
from test_artifact_audit import ORPHAN, Store
from test_backup_archive import backup as backup
from test_backup_archive import repin

from cannery.maintenance.archive import verify
from cannery.maintenance.collect import collect


def prepare(backup, data=b"orphan"):
    directory, manifest = backup
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        member = tarfile.TarInfo(ORPHAN)
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))
    repin(directory, manifest, "artifacts", output.getvalue())

    class Collector(Store):
        deleted = False

        def volume(self, image, name, program, *, writable=False, **kwargs):
            if writable:
                self.deleted = True
                assert "unlink_verified" in program
                return b""
            return super().volume(image, name, program, **kwargs)

    docker = Collector()
    docker.rows[-1].update(size=len(data), sha256=hashlib.sha256(data).hexdigest())
    # Real content-addressed orphan name, with matching archived bytes.
    digest = docker.rows[-1]["sha256"]
    docker.rows[-1]["name"] = digest
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        member = tarfile.TarInfo(digest)
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))
    repin(directory, verify(directory), "artifacts", output.getvalue())
    return directory, docker


def test_deletion_requires_recoverable_bytes_and_no_live_errors(backup):
    directory, docker = prepare(backup)
    result = collect(docker, "source", directory, 24)
    assert docker.deleted and result["deleted"] == [hashlib.sha256(b"orphan").hexdigest()]


@pytest.mark.parametrize("fault", ["missing", "corrupt", "foreign", "live"])
def test_collection_fails_before_deletion_when_backup_or_live_state_is_unsafe(backup, fault):
    directory, docker = prepare(backup)
    if fault == "missing":
        docker.rows[-1]["name"] = "f" * 64
        docker.rows[-1]["sha256"] = "f" * 64
    elif fault == "corrupt":
        (directory / "artifact-data.tar").write_bytes(b"bad")
    elif fault == "live":
        docker.rows[0]["sha256"] = "f" * 64
    with pytest.raises(ValueError):
        collect(docker, "other" if fault == "foreign" else "source", directory, 24)
    assert not docker.deleted
