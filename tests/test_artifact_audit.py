import json
import time

import pytest
from test_backup_capture import Fake

from cannery.maintenance.artifacts import audit, references

LIVE = "a" * 64
PENDING = "b" * 64
ORPHAN = "c" * 64


class Store(Fake):
    def __init__(self):
        super().__init__()
        self.rows = [
            dict(name=sha, sha256=sha, size=1, mtime_ns=0, inode=i)
            for i, sha in enumerate([LIVE, PENDING, ORPHAN])
        ]
        self.records = [
            {"artifact": dict(repository="private", revision=LIVE, sha256=LIVE, path="log")},
            {"unknown_future_reference": PENDING},
        ]

    def run(self, *args, **kwargs):
        if "FROM pg_tables" in args[-1]:
            return b'["results"]'
        if "row_to_json" in args[-1]:
            return "\n".join(json.dumps(r) for r in self.records).encode()
        return super().run(*args, **kwargs)

    def volume(self, image, name, program, **kwargs):
        return json.dumps(
            self.rows if name == "blobs" else {"protected": [], "required": []}
        ).encode()


def test_all_history_and_future_sha_fields_are_conservatively_protected():
    result = audit(Store(), "source", 24)
    assert result["errors"] == [] and result["required"] == 1 and result["protected"] == 2
    assert [r["name"] for r in result["candidates"]] == [ORPHAN]


def test_missing_or_corrupt_referenced_bytes_block_safe_collection():
    store = Store()
    store.rows = store.rows[1:]
    assert "missing:" + LIVE in audit(store, "source")["errors"]
    store = Store()
    store.rows[0]["sha256"] = "f" * 64
    assert "corrupt:" + LIVE in audit(store, "source")["errors"]


def test_young_uploads_unknown_files_and_temporary_files_have_explicit_outcomes():
    store = Store()
    store.rows[2]["mtime_ns"] = time.time_ns()
    store.rows.append(dict(name="tmpabcdefgh", sha256="d" * 64, size=3, mtime_ns=0, inode=4))
    assert [r["name"] for r in audit(store, "source")["candidates"]] == ["tmpabcdefgh"]
    store.rows.append(dict(name="operator-note", sha256="e" * 64, size=1, mtime_ns=0, inode=5))
    assert audit(store, "source")["errors"] == ["unknown:operator-note"]


def test_journal_pins_are_protected_even_before_sql_completion():
    class Journal(Store):
        def volume(self, image, name, program, **kwargs):
            if name == "builds":
                return json.dumps({"protected": [ORPHAN], "required": [ORPHAN]}).encode()
            return super().volume(image, name, program, **kwargs)

    assert audit(Journal(), "source")["candidates"] == []


def test_invalid_refs_network_failure_and_running_writers_fail_closed():
    with pytest.raises(ValueError):
        references(dict(repository="private", revision="bad", path="x", sha256=LIVE))
    store = Store()
    store.containers[1]["State"] = {"Running": True, "Status": "running"}
    with pytest.raises(ValueError):
        audit(store, "source")
    with pytest.raises(ValueError):
        audit(Store(), "source", 0)

    class Offline(Store):
        def volume(self, *args, **kwargs):
            raise ValueError("Disconnected")

    with pytest.raises(ValueError):
        audit(Offline(), "source")


@pytest.mark.parametrize("nested", [False, True])
def test_journal_enumeration_errors_abort(monkeypatch, tmp_path, nested):
    import os

    from cannery.maintenance.artifacts import scan_journals

    (tmp_path / "nested").mkdir()
    original_walk, original_scandir = os.walk, os.scandir
    failed = tmp_path / "nested" if nested else tmp_path

    def scandir(path):
        if str(path) == str(failed):
            raise OSError("injected enumeration failure")
        return original_scandir(path)

    monkeypatch.setattr(os, "walk", lambda root, **kwargs: original_walk(tmp_path, **kwargs))
    monkeypatch.setattr(os, "scandir", scandir)
    with pytest.raises(OSError, match="injected"):
        scan_journals()


def test_oversized_json_journal_aborts_reference_scan(monkeypatch, tmp_path):
    import os

    from cannery.maintenance.artifacts import scan_journals

    reference = dict(repository="private", revision=LIVE, path="missing", sha256=LIVE)
    (tmp_path / "receipt.json").write_text(json.dumps(dict(ref=reference, padding="x" * 262144)))
    original_walk = os.walk
    monkeypatch.setattr(os, "walk", lambda root, **kwargs: original_walk(tmp_path, **kwargs))
    with pytest.raises(ValueError, match="JSON exceeds"):
        scan_journals()
