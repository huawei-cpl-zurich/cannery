"""Conservative offline artifact reachability: retained SQL history and controller journals."""

import hashlib
import inspect
import json
import os
import re
import stat
import time
from pathlib import Path
from typing import Any

from cannery.maintenance.capture import Docker, inventory

TOKEN = re.compile(r"(?<![a-f0-9])[a-f0-9]{64}(?![a-f0-9])")


def references(value: Any) -> tuple[set[str], set[str]]:
    protected: set[str] = set()
    required: set[str] = set()
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            if (
                item.get("repository") == "private"
                and {"revision", "sha256", "path"} <= item.keys()
            ):
                sha = item["sha256"]
                if (
                    not isinstance(sha, str)
                    or re.fullmatch(r"[a-f0-9]{64}", sha) is None
                    or item["revision"] != sha
                ):
                    raise ValueError("Invalid retained private artifact reference")
                required.add(sha)
            pending.extend(item.values())
            pending.extend(item.keys())
        elif isinstance(item, list):
            pending.extend(item)
        elif isinstance(item, str):
            protected.update(TOKEN.findall(item))
    return protected, required


def scan_journals() -> dict[str, list[str]]:
    protected: set[str] = set()
    required: set[str] = set()
    total = count = 0

    def fail(error: OSError) -> None:
        raise error

    for directory, dirs, files in os.walk("/data", followlinks=False, onerror=fail):
        for name in dirs:
            if (Path(directory) / name).is_symlink():
                raise ValueError("Symlink in controller journal")
        for name in files:
            path = Path(directory) / name
            if not stat.S_ISREG(path.lstat().st_mode):
                raise ValueError("Nonregular journal file")
            count += 1
            total += path.stat().st_size
            if count > 10000 or total > 1024**3:
                raise ValueError("Journal scan exceeds budget")
            data = bytearray()
            tail = b""
            with path.open("rb") as stream:
                while chunk := stream.read(65536):
                    protected.update(m.decode() for m in re.findall(rb"[a-f0-9]{64}", tail + chunk))
                    tail = chunk[-64:]
                    if len(data) <= 256 * 1024:
                        data.extend(chunk)
            if path.suffix == ".json":
                if len(data) > 256 * 1024:
                    raise ValueError("Journal JSON exceeds parsing budget")
                values = references(json.loads(data))
                protected.update(values[0])
                required.update(values[1])
    return {"protected": sorted(protected), "required": sorted(required)}


def scan_blobs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(Path("/data").iterdir()):
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("Artifact store contains nonregular entries")
        if len(rows) >= 10000 or info.st_size > 64 * 1024 * 1024:
            raise ValueError("Artifact scan exceeds budget")
        sha = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                sha.update(chunk)
        rows.append(
            dict(
                name=path.name,
                sha256=sha.hexdigest(),
                size=info.st_size,
                mtime_ns=info.st_mtime_ns,
                inode=info.st_ino,
            )
        )
    return rows


def program(function: Any) -> str:
    return (
        "from __future__ import annotations\nimport os,stat,json,re,hashlib\n"
        "from pathlib import Path\nfrom typing import Any\n"
        + 'TOKEN=re.compile(r"(?<![a-f0-9])[a-f0-9]{64}(?![a-f0-9])")\n'
        + inspect.getsource(references)
        + "\n"
        + inspect.getsource(function)
        + f"\nprint(json.dumps({function.__name__}()))"
    )


def audit(docker: Docker, name: str, age_hours: float = 168) -> dict[str, Any]:
    if not 24 <= age_hours <= 24 * 3650:
        raise ValueError("Minimum artifact age must be at least 24 hours")
    project = inventory(docker, name)
    deadline = time.monotonic() + 300
    protected: set[str] = set()
    required: set[str] = set()
    tables = json.loads(
        project.sql(
            "SELECT COALESCE(json_agg(tablename),'[]'::json) FROM pg_tables "
            "WHERE schemaname='public'"
        )
    )
    if len(tables) > 1000:
        raise ValueError("Database table inventory exceeds budget")
    records = 0
    for table in tables:
        quoted = '"' + table.replace('"', '""') + '"'
        offset = 0
        while True:
            if time.monotonic() > deadline:
                raise ValueError("Artifact mark deadline exceeded")
            rows = project.sql(
                f"SELECT row_to_json(t) FROM {quoted} t ORDER BY t.tableoid,t.ctid "
                f"LIMIT 100 OFFSET {offset}"
            ).splitlines()
            for row in rows:
                values = references(json.loads(row))
                protected.update(values[0])
                required.update(values[1])
            records += len(rows)
            if records > 100000:
                raise ValueError("Database mark scan exceeds budget")
            if len(rows) < 100:
                break
            offset += 100
    for role in ["build_control", "baseline_control"]:
        if role in project.volumes:
            data = json.loads(
                docker.volume(project.helper_image, project.volumes[role], program(scan_journals))
            )
            protected.update(data["protected"])
            required.update(data["required"])
    rows = json.loads(
        docker.volume(project.helper_image, project.volumes["artifacts"], program(scan_blobs))
    )
    present = {row["name"]: row for row in rows}
    errors = ["missing:" + sha for sha in sorted(required - present.keys())]
    cutoff = time.time_ns() - int(age_hours * 3600 * 10**9)
    candidates = []
    for row in rows:
        if re.fullmatch(r"[a-f0-9]{64}", row["name"]):
            if row["name"] != row["sha256"]:
                errors.append("corrupt:" + row["name"])
                continue
            if row["name"] in protected:
                continue
        elif re.fullmatch(r"tmp[a-z0-9_]{8}", row["name"]) is None:
            errors.append("unknown:" + row["name"])
            continue
        if row["mtime_ns"] <= cutoff:
            candidates.append(row)
    if inventory(docker, name) != project:
        raise ValueError("Project changed during artifact audit")
    return {
        "project": name,
        "protected": len(protected),
        "required": len(required),
        "objects": len(rows),
        "candidates": candidates,
        "errors": sorted(errors),
        "minimum_age_hours": age_hours,
    }
