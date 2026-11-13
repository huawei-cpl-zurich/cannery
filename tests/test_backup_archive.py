import hashlib
import io
import json
import os
import tarfile
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from cannery.maintenance.archive import BackupManifest, checksum, verify


def tar(entries):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        for name, kind in entries:
            info = tarfile.TarInfo(name)
            info.type = kind
            info.mode = 0o700
            info.linkname = "/tmp/outside" if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE) else ""
            archive.addfile(info, io.BytesIO())
    return output.getvalue()


@pytest.fixture
def backup(tmp_path):
    roles = {
        "database": ("database.dump", b"PGDMPdata"),
        "snapshots": ("git-snapshots.tar", tar([(".", tarfile.DIRTYPE)])),
        "artifacts": ("artifact-data.tar", tar([("blob", tarfile.REGTYPE)])),
    }
    files = []
    for role, (name, data) in roles.items():
        (tmp_path / name).write_bytes(data)
        files.append(dict(role=role, size=len(data), sha256=hashlib.sha256(data).hexdigest()))
    manifest = BackupManifest.model_validate_json(
        json.dumps(
            {
                "id": str(uuid4()),
                "created_at": datetime.now(UTC).isoformat(),
                "project": "source",
                "postgres_version": 170000,
                "migration": "0013",
                "images": {"api": "sha256:" + "a" * 64},
                "files": files,
            }
        )
    )
    (tmp_path / "manifest.json").write_bytes(manifest.encoded())
    return tmp_path, manifest


def repin(directory, manifest, role, data):
    file = next(f for f in manifest.files if f.role == role)
    (directory / file.filename).write_bytes(data)
    document = manifest.model_dump(mode="json")
    for item in document["files"]:
        if item["role"] == role:
            item.update(size=len(data), sha256=hashlib.sha256(data).hexdigest())
    (directory / "manifest.json").write_bytes(
        BackupManifest.model_validate_json(json.dumps(document)).encoded()
    )


def test_valid_backup_and_streaming_digest(backup):
    directory, manifest = backup
    assert verify(directory) == manifest
    data = b"x" * (2 * 1024 * 1024 + 1)
    (directory / "large").write_bytes(data)
    assert checksum(directory / "large") == (len(data), hashlib.sha256(data).hexdigest())


@pytest.mark.parametrize("fault", ["corrupt", "truncated", "missing", "extra", "symlink", "fifo"])
def test_incomplete_or_modified_backup_is_rejected(backup, fault, tmp_path):
    directory, _ = backup
    path = directory / "database.dump"
    if fault == "extra":
        (directory / "unexpected").write_bytes(b"x")
    elif fault == "corrupt":
        path.write_bytes(b"bad")
    elif fault == "truncated":
        path.write_bytes(path.read_bytes()[:-1])
    else:
        path.unlink()
        if fault == "symlink":
            path.symlink_to(directory / "manifest.json")
        elif fault == "fifo":
            os.mkfifo(path)
    with pytest.raises((ValueError, OSError)):
        verify(directory)


@pytest.mark.parametrize(
    "name,kind",
    [
        ("../escape", tarfile.REGTYPE),
        ("/absolute", tarfile.REGTYPE),
        ("nested/../../escape", tarfile.REGTYPE),
        ("back\\slash", tarfile.REGTYPE),
        ("link", tarfile.SYMTYPE),
        ("hardlink", tarfile.LNKTYPE),
        ("fifo", tarfile.FIFOTYPE),
        ("device", tarfile.CHRTYPE),
    ],
)
def test_unsafe_archive_is_rejected_even_with_matching_checksum(backup, name, kind):
    directory, manifest = backup
    repin(directory, manifest, "snapshots", tar([(name, kind)]))
    with pytest.raises(ValueError, match="Unsafe"):
        verify(directory)


def test_duplicate_members_and_wrong_database_format_are_rejected(backup):
    directory, manifest = backup
    repin(directory, manifest, "artifacts", tar([("blob", tarfile.REGTYPE)] * 2))
    with pytest.raises(ValueError):
        verify(directory)
    repin(directory, manifest, "artifacts", tar([("blob", tarfile.REGTYPE)]))
    manifest = verify(directory)
    repin(directory, manifest, "database", b"plain SQL is unsupported")
    with pytest.raises(ValueError, match="custom-format"):
        verify(directory)


@pytest.mark.parametrize(
    "change",
    [
        {"project": "../bad"},
        {"created_at": "2026-01-01T00:00:00"},
        {"images": {"api": "latest"}},
        {"postgres_version": 160000},
    ],
)
def test_manifest_rejects_ambiguous_identity_or_unsupported_database(backup, change):
    _, manifest = backup
    with pytest.raises(ValueError):
        BackupManifest.model_validate_json(json.dumps(manifest.model_dump(mode="json") | change))


@pytest.mark.parametrize("kind", [tarfile.XHDTYPE, tarfile.GNUTYPE_LONGNAME])
def test_extension_header_is_bounded_before_payload_read(backup, kind):
    directory, manifest = backup
    header = tarfile.TarInfo("metadata")
    header.type, header.size = kind, 32768
    data = header.tobuf() + bytes(32768 + 1024)
    repin(directory, manifest, "snapshots", data)
    with pytest.raises(ValueError, match="metadata exceeds budget"):
        verify(directory)


def test_recursive_and_sparse_extension_metadata_is_rejected(backup):
    directory, manifest = backup
    header = tarfile.TarInfo("metadata")
    header.type, header.size = tarfile.XHDTYPE, 0
    repin(directory, manifest, "snapshots", header.tobuf() * 17 + bytes(1024))
    with pytest.raises(ValueError, match="metadata exceeds budget"):
        verify(directory)
    header.size = 512
    repin(
        directory,
        manifest,
        "snapshots",
        header.tobuf() + b"GNU.sparse" + bytes(512 - len(b"GNU.sparse")) + bytes(1024),
    )
    with pytest.raises(ValueError, match="Sparse metadata"):
        verify(directory)


def test_tar_end_marker_and_padding_are_verified(backup):
    directory, manifest = backup
    data = tar([("file", tarfile.REGTYPE)])
    repin(directory, manifest, "snapshots", data[:512])
    with pytest.raises(ValueError):
        verify(directory)
    repin(directory, manifest, "snapshots", data + b"hidden")
    with pytest.raises(ValueError, match="after tar end"):
        verify(directory)


@pytest.mark.parametrize("format", [tarfile.PAX_FORMAT, tarfile.GNU_FORMAT])
def test_bounded_long_name_extensions_remain_supported(backup, format):
    directory, manifest = backup
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=format) as archive:
        archive.addfile(tarfile.TarInfo("nested/" + "a" * 180), io.BytesIO())
    repin(directory, manifest, "snapshots", output.getvalue())
    assert verify(directory).project == "source"


@pytest.mark.parametrize(
    "entries",
    [
        [("ok", tarfile.REGTYPE), ("bad\0name", tarfile.REGTYPE)],
        [("parent", tarfile.REGTYPE), ("parent/child", tarfile.REGTYPE)],
        [("parent/child", tarfile.REGTYPE), ("parent", tarfile.REGTYPE)],
        [(".", tarfile.REGTYPE)],
    ],
)
def test_unrestorable_paths_are_rejected_before_extraction(backup, entries):
    directory, manifest = backup
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for name, kind in entries:
            info = tarfile.TarInfo(name)
            info.type = kind
            if "\0" in name:
                info.name = "safe"
                info.pax_headers = {"path": name}
            archive.addfile(info, io.BytesIO())
    repin(directory, manifest, "snapshots", output.getvalue())
    with pytest.raises(ValueError):
        verify(directory)


def test_pax_size_cannot_hide_an_unbounded_extension_header(backup):
    directory, manifest = backup
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.PAX_FORMAT) as archive:
        hidden = tarfile.TarInfo("hidden")
        hidden.type, hidden.size = tarfile.XHDTYPE, 32768
        payload = hidden.tobuf() + bytes(32768 + 512)
        outer = tarfile.TarInfo("outer")
        outer.size = len(payload)
        outer.pax_headers = {"size": "0"}
        archive.addfile(outer, io.BytesIO(payload))
    repin(directory, manifest, "snapshots", output.getvalue())
    with pytest.raises(ValueError, match="Unsupported PAX key"):
        verify(directory)
