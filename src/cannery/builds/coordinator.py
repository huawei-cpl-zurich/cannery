"""Trusted coordinator; author compilation/native loading happens only in job containers."""

import argparse
import json
import logging
import threading
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import Engine, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from cannery.artifacts import ArtifactError
from cannery.artifacts.factory import make_store
from cannery.builds.ascendc import BuildError
from cannery.builds.executor import BuildExecutor, ExecutionFailure
from cannery.builds.remote import (
    CplRemote,
    RemoteFailure,
    RemotePending,
    durable_write,
    operation_receipts,
    settle_operation,
)
from cannery.builds.sandbox import WorkerProfile
from cannery.builds.state import (
    BuildAttempt,
    BuildClaim,
    BuildResult,
    BuildState,
    claim_build,
    current_build,
    due_builds,
    finish_build,
    heartbeat,
    queue_build,
)
from cannery.database import make_engine
from cannery.devices import Device
from cannery.git_intake.snapshots import SnapshotStore
from cannery.settings import Settings
from cannery.shutdown import stop_on_term
from cannery.submissions.models import Candidate, SubmissionReuse


def discover(engine: Engine, *, device: Device = "a5") -> None:
    with Session(engine) as session, session.begin():
        candidates = session.scalars(
            select(Candidate)
            .where(
                Candidate.manifest["implementation"]["device"].as_string() == device,
                Candidate.manifest["implementation"]["build"]["abi"]
                .as_string()
                .in_(("unary-f32-v1", "catlass-unary-f32-v1")),
                ~select(BuildState.id).where(BuildState.id == Candidate.id).exists(),
                ~select(SubmissionReuse.id).where(SubmissionReuse.id == Candidate.id).exists(),
            )
            .order_by(Candidate.id)
            .limit(50)
            .with_for_update(skip_locked=True)
        )
        for candidate in candidates:
            if not queue_build(session, candidate):
                session.add(
                    BuildState(id=candidate.id, status="failed", error="unsupported_build_contract")
                )


def postpone(engine: Engine, claim: BuildClaim) -> None:
    with Session(engine) as session, session.begin():
        state = current_build(session, claim)
        if state is not None:
            state.lease = None
            state.due_at = datetime.now(UTC) + timedelta(seconds=5)
            state.lease_until = state.due_at


class LeaseKeeper:
    def __init__(self, engine: Engine, claim: BuildClaim) -> None:
        self.engine, self.claim = engine, claim
        self.stop, self.lost = threading.Event(), threading.Event()
        self.thread = threading.Thread(target=self.maintain, daemon=True)

    def maintain(self) -> None:
        while not self.stop.wait(20):
            try:
                if not heartbeat(self.engine, self.claim):
                    self.lost.set()
                    return
            except SQLAlchemyError:
                self.lost.set()
                return

    def fence(self) -> None:
        with Session(self.engine) as session:
            if self.lost.is_set() or current_build(session, self.claim) is None:
                raise RemotePending("Coordinator no longer owns the build")

    def __enter__(self) -> "LeaseKeeper":
        self.thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop.set()
        self.thread.join(timeout=2)


def evidence(executor: BuildExecutor) -> dict[str, Any]:
    journal = executor.remote.journal
    context = journal / "context.json"
    frozen = json.loads(context.read_bytes()) if context.exists() else {}
    details: dict[str, Any] = {
        "worker": executor.profile.model_dump(mode="json"),
        "loader_sha256": frozen.get("wheel", executor.wheel_sha256),
        "operations": {},
        "logs": {},
    }
    receipts = operation_receipts(journal)
    native = sorted(
        (p for p in receipts if p.stem.startswith("runtime-")),
        key=lambda p: int(p.stem.split("-")[1]),
    )
    chosen = {"compile"} | ({native[-1].stem} if native else set())
    for receipt in receipts:
        data = json.loads(receipt.read_bytes())
        if "handle" not in data:
            continue
        details["operations"][receipt.stem] = data["handle"]
        if receipt.stem not in chosen:
            continue
        logs = {
            stream: executor.remote.call(
                "logs", data["handle"], "--stream", stream, "--tail", "200"
            ).get("content", "")[-16384:]
            for stream in ("stdout", "stderr")
        }
        reference = executor.artifacts.put(
            f"builds/{journal.name}/{receipt.stem}-log.json", json.dumps(logs).encode()
        )
        details["logs"][receipt.stem] = reference.model_dump(mode="json")
    durable_write(journal / "evidence.json", json.dumps(details, sort_keys=True).encode())
    return details


def process(
    engine: Engine,
    key: UUID,
    factory: Callable[[BuildClaim], BuildExecutor],
    *,
    device: Device = "a5",
) -> None:
    claim = claim_build(engine, key, device=device)
    if claim is None:
        return
    with LeaseKeeper(engine, claim) as keeper:
        try:
            executor = factory(claim)
            with Session(engine) as session:
                candidate = session.get(Candidate, key)
                if candidate is None:
                    raise ExecutionFailure("retained_source_missing")
            try:
                reference = executor.execute(candidate, claim.attempt, keeper.fence)
            except (BuildError, RemoteFailure) as failure:
                if list(executor.remote.journal.glob("*.dispatch")):
                    executor.cleanup(claim.attempt)
                details = evidence(executor)
                keeper.fence()
                finished = finish_build(
                    engine,
                    claim,
                    error=failure.code
                    if isinstance(failure, ExecutionFailure)
                    else (
                        "unsupported_build_contract"
                        if isinstance(failure, BuildError)
                        else "worker_failed"
                    ),
                    retryable=failure.retryable
                    if isinstance(failure, ExecutionFailure)
                    else not isinstance(failure, BuildError),
                    evidence=details,
                )
                if finished:
                    prune_cache(executor.remote.journal)
                return
            executor.cleanup(claim.attempt)
            details = evidence(executor)
            keeper.fence()
            if finish_build(engine, claim, artifact=reference, evidence=details):
                prune_cache(executor.remote.journal)
        except RemotePending:
            postpone(engine, claim)
        except ArtifactError:
            logging.warning("Private artifact storage unavailable: %s", key)
            postpone(engine, claim)
        except (OSError, ValueError, SQLAlchemyError):
            logging.warning("Binary admission unavailable: %s", key)
            postpone(engine, claim)


def prune_cache(journal: Path) -> None:
    for name in ("source.tar.gz", "runtime.tar.gz", "kernel.so"):
        (journal / name).unlink(missing_ok=True)
    durable_write(journal / "cleanup-complete", b"complete\n")


def cleanup_finished(
    engine: Engine,
    root: Path,
    factory: Callable[[BuildClaim], BuildExecutor],
) -> None:
    if not root.exists():
        return
    count = 0
    cursor = root / "cleanup-cursor"
    previous = cursor.read_text() if cursor.exists() else ""
    journals = sorted(root.iterdir(), key=lambda path: path.name)
    journals = [path for path in journals if path.name > previous] + [
        path for path in journals if path.name <= previous
    ]
    for journal in journals:
        if not journal.is_dir() or journal.is_symlink() or (journal / "cleanup-complete").exists():
            continue
        try:
            identifier = UUID(hex=journal.name)
        except ValueError:
            continue
        with Session(engine) as session:
            attempt = session.get(BuildAttempt, identifier)
            if attempt is None or session.get(BuildResult, identifier) is None:
                continue
            claim = BuildClaim(attempt.candidate_id, identifier, identifier)
        if count >= 5:
            break
        count += 1
        durable_write(cursor, journal.name.encode())
        try:
            executor = factory(claim)
            context = journal / "context.json"
            if context.exists():
                executor.profile = WorkerProfile.model_validate_json(
                    json.dumps(json.loads(context.read_bytes())["profile"])
                )
                executor.remote.target = executor.profile.target
            for receipt in operation_receipts(journal):
                saved = json.loads(receipt.read_bytes())
                if "handle" in saved:
                    settle_operation(executor.remote, saved["handle"], cancel_jobs=True)
            if list(journal.glob("*.dispatch")):
                executor.cleanup(identifier)
            evidence(executor)
            prune_cache(journal)
        except (RemotePending, RemoteFailure, ArtifactError, OSError, ValueError):
            logging.warning("Retaining unfinished build cleanup: %s", identifier)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    settings = Settings()
    if (
        settings.artifact_backend != "private"
        or settings.build_profile is None
        or settings.build_loader_wheel is None
        or settings.build_loader_sha256 is None
    ):
        raise SystemExit("Configure private storage, build profile and pinned loader wheel first")
    profile = WorkerProfile.model_validate_json(settings.build_profile.read_bytes())
    wheel = settings.build_loader_wheel
    wheel_sha256 = settings.build_loader_sha256
    store = make_store(settings)
    engine = make_engine(settings)

    def factory(claim: BuildClaim) -> BuildExecutor:
        remote = CplRemote(
            settings.build_remote_client,
            settings.build_control_root / claim.attempt.hex,
            profile.target,
            registry=settings.build_remote_registry,
        )
        return BuildExecutor(
            profile,
            remote,
            SnapshotStore(settings.snapshot_root),
            store,
            wheel,
            wheel_sha256,
        )

    stopped = threading.Event()
    restore_signals = stop_on_term(stopped)

    def cleanup_loop() -> None:
        while not stopped.is_set():
            try:
                cleanup_finished(engine, settings.build_control_root, factory)
            except (SQLAlchemyError, OSError):
                logging.warning("Build cleanup unavailable; retaining journals")
            stopped.wait(10)

    cleaner = threading.Thread(target=cleanup_loop, daemon=True)
    if not args.once:
        cleaner.start()
    try:
        while not stopped.is_set():
            try:
                discover(engine, device=profile.device)
                for key in due_builds(engine, device=profile.device):
                    if stopped.is_set():
                        break
                    process(engine, UUID(key), factory, device=profile.device)
            except (SQLAlchemyError, OSError, ValueError):
                logging.warning("Binary admission coordinator unavailable; retrying")
            if args.once:
                cleanup_finished(engine, settings.build_control_root, factory)
                return
            stopped.wait(5)
    finally:
        restore_signals()
        stopped.set()
        if cleaner.is_alive():
            cleaner.join(timeout=2)
        engine.dispose()


if __name__ == "__main__":
    main()
