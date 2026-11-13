"""Resume fenced performance attempts and finalize only after owned cleanup."""

import json
import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from cannery.artifacts import ArtifactError, ArtifactRef
from cannery.benchmark.executor import BenchmarkExecutor
from cannery.benchmark.scheduling import BenchmarkBlocked, ReferencePending, retained
from cannery.benchmark.state import (
    PerformanceClaim,
    PerformanceJob,
    PerformanceResult,
    claim_job,
    current_job,
    finish_job,
    heartbeat,
    postpone_job,
)
from cannery.builds.coordinator import evidence
from cannery.builds.executor import ExecutionFailure
from cannery.builds.remote import (
    RemoteFailure,
    RemotePending,
    durable_write,
    operation_receipts,
    settle_operation,
)
from cannery.builds.sandbox import WorkerProfile
from cannery.numerics.baselines import BaselinePending, BaselineUnavailable


class LeaseKeeper:
    def __init__(self, engine: Engine, claim: PerformanceClaim) -> None:
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
        with Session(self.engine) as db:
            if self.lost.is_set() or current_job(db, self.claim) is None:
                raise RemotePending("Performance lease lost")

    def __enter__(self) -> "LeaseKeeper":
        self.thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop.set()
        self.thread.join(timeout=2)


def prune(journal: Path) -> None:
    for name in ("benchmark.tar.gz", "results.json"):
        (journal / name).unlink(missing_ok=True)
    durable_write(journal / "cleanup-complete", b"complete\n")


def process(
    engine: Engine, key: str, factory: Callable[[PerformanceClaim], BenchmarkExecutor]
) -> None:
    claim = claim_job(engine, key)
    if claim is None:
        return
    with LeaseKeeper(engine, claim) as keeper:
        try:
            executor = factory(claim)
            executor.fence = keeper.fence
            failure_path = executor.remote.journal / "benchmark-failure.json"
            outcome: dict[str, Any] = {}
            report = None
            artifact = None
            if failure_path.exists():
                saved = json.loads(failure_path.read_bytes())
                outcome = saved["outcome"]
                report = saved["report"]
                artifact = (
                    ArtifactRef.model_validate(saved["artifact"]) if saved["artifact"] else None
                )
            else:
                try:
                    with Session(engine) as db:
                        job = db.get(PerformanceJob, key)
                        assert job is not None
                        data = retained(engine, job, executor)
                    report, artifact = executor.benchmark(claim.attempt, keeper.fence, **data)
                    if not report["passed"]:
                        outcome = {
                            "error": "benchmark_correctness_failed",
                            "blocked": False,
                            "retryable": False,
                        }
                except (
                    ExecutionFailure,
                    RemoteFailure,
                    BenchmarkBlocked,
                    BaselineUnavailable,
                ) as failure:
                    outcome = {
                        "error": failure.code
                        if isinstance(failure, ExecutionFailure)
                        else str(failure)
                        if isinstance(failure, (BenchmarkBlocked, BaselineUnavailable))
                        else "benchmark_worker_failed",
                        "blocked": isinstance(failure, (BenchmarkBlocked, BaselineUnavailable)),
                        "retryable": failure.retryable
                        if isinstance(failure, ExecutionFailure)
                        else isinstance(failure, RemoteFailure),
                    }
                if outcome:
                    keeper.fence()
                    durable_write(
                        failure_path,
                        json.dumps(
                            {
                                "outcome": outcome,
                                "report": report,
                                "artifact": artifact.model_dump(mode="json") if artifact else None,
                            },
                            sort_keys=True,
                        ).encode(),
                    )
            keeper.fence()
            if list(executor.remote.journal.glob("*.dispatch")):
                executor.cleanup(claim.attempt)
            details = evidence(executor)
            report = (report or {}) | {"evidence": details}
            keeper.fence()
            if finish_job(engine, claim, report=report, artifact=artifact, **outcome):
                prune(executor.remote.journal)
        except (
            RemotePending,
            ReferencePending,
            BaselinePending,
            ArtifactError,
            OSError,
            ValueError,
            SQLAlchemyError,
        ) as pending:
            logging.warning("Performance operation pending: %s", key)
            reason = (
                "artifact_storage_unavailable"
                if isinstance(pending, ArtifactError)
                else "awaiting_reference_timing"
                if isinstance(pending, ReferencePending)
                else "baseline_storage_unavailable"
                if isinstance(pending, BaselinePending)
                else "worker_control_unavailable"
                if isinstance(pending, OSError)
                else "database_unavailable"
                if isinstance(pending, SQLAlchemyError)
                else "retained_context_unavailable"
                if isinstance(pending, ValueError)
                else None
            )
            postpone_job(engine, claim, reason=reason)


def cleanup_finished(
    engine: Engine, root: Path, factory: Callable[[PerformanceClaim], BenchmarkExecutor]
) -> None:
    if not root.exists():
        return
    cursor = root / "cleanup-cursor"
    previous = cursor.read_text() if cursor.exists() else ""
    journals = sorted(root.iterdir(), key=lambda p: p.name)
    journals = [p for p in journals if p.name > previous] + [
        p for p in journals if p.name <= previous
    ]
    count = 0
    for journal in journals:
        if not journal.is_dir() or journal.is_symlink() or (journal / "cleanup-complete").exists():
            continue
        try:
            identifier = UUID(hex=journal.name)
        except ValueError:
            continue
        with Session(engine) as db:
            result = db.get(PerformanceResult, identifier)
        if result is None:
            continue
        if count >= 5:
            break
        count += 1
        durable_write(cursor, journal.name.encode())
        try:
            executor = factory(PerformanceClaim(result.job_key, result.id, result.id))
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
                executor.cleanup(result.id)
            evidence(executor)
            prune(journal)
        except (RemotePending, RemoteFailure, ArtifactError, OSError, ValueError):
            logging.warning("Retaining performance cleanup: %s", result.id)
