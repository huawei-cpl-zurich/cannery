"""Bounded trusted hardware identity discovery; retained probes survive restarts."""

import fcntl
import json
import re
import shutil
import time
from pathlib import Path
from uuid import UUID, uuid4

from cannery.benchmark.contract import Cohort
from cannery.builds.remote import CplRemote, RemoteFailure, RemotePending, durable_write
from cannery.builds.sandbox import WorkerProfile, job_script
from cannery.registry.manifest import unique_pairs


def prune_completed(root: Path) -> None:
    completed = sorted(
        (p for p in root.iterdir() if p.is_dir() and not p.is_symlink() and (p / "done").exists()),
        key=lambda p: p.stat().st_mtime,
    )
    for old in completed[:-2]:
        try:
            UUID(hex=old.name)
        except ValueError:
            continue
        shutil.rmtree(old)


def device_cohort(
    profile: WorkerProfile, root: Path, client: Path, registry: Path, loader_sha256: str
) -> Cohort:
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (root / "lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RemotePending("Device identity probe owned by another worker") from None
        cache = root / "identity.json"
        pending = root / "pending.json"
        now = time.time()
        current = profile.model_dump(mode="json")
        if cache.exists():
            saved = json.loads(cache.read_bytes())
            if (
                saved["profile"] == current
                and saved["loader"] == loader_sha256
                and 0 <= now - saved["at"] < 60
            ):
                return Cohort.model_validate_json(json.dumps(saved["cohort"]))
        if not pending.exists():
            durable_write(
                pending,
                json.dumps(
                    {"profile": current, "attempt": uuid4().hex, "loader": loader_sha256}
                ).encode(),
            )
        saved = json.loads(pending.read_bytes())
        pinned = WorkerProfile.model_validate_json(json.dumps(saved["profile"]))
        attempt = UUID(hex=saved["attempt"])
        remote = CplRemote(client, root / attempt.hex, pinned.target, registry=registry)
        try:
            handle = remote.operation(
                "identify",
                job_script(pinned, attempt, "identify"),
                workspace=pinned.workspace,
                runtime=pinned.runtime,
            )
        except RemoteFailure:
            # Confirmed terminal probe failure: a later poll may make a new read-only probe.
            durable_write(remote.journal / "done", b"terminal-failure\n")
            pending.unlink()
            prune_completed(root)
            raise RemotePending("Device identity unavailable") from None
        reply = remote.call("logs", handle, "--tail", "2")
        identity = json.loads(reply.get("content", ""), object_pairs_hook=unique_pairs)
        if (
            not isinstance(identity, dict)
            or set(identity) != {"hardware_id", "driver"}
            or any(
                not isinstance(v, str) or re.fullmatch(r"[a-f0-9]{64}", v) is None
                for v in identity.values()
            )
        ):
            raise ValueError("Invalid trusted device identity")
        cohort = Cohort(
            **identity,
            device=pinned.device,
            image=pinned.image,
            loader_sha256=saved["loader"],
            runtime=pinned.fingerprint,
        )
        durable_write(
            cache,
            json.dumps(
                {
                    "profile": saved["profile"],
                    "loader": saved["loader"],
                    "at": now,
                    "cohort": cohort.model_dump(mode="json"),
                }
            ).encode(),
        )
        durable_write(remote.journal / "done", b"complete\n")
        pending.unlink()
        prune_completed(root)
        if saved["profile"] != current or saved["loader"] != loader_sha256:
            raise RemotePending("Previous device probe reconciled; current profile needs discovery")
        return cohort
