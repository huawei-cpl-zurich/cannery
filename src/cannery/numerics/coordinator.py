"""Trusted coordinator for post-admission numerical validation."""

import json
import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import Engine, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from cannery.artifacts import ArtifactError, ArtifactRef
from cannery.builds.coordinator import evidence
from cannery.builds.executor import BuildExecutor, ExecutionFailure
from cannery.builds.remote import (
    RemoteFailure,
    RemotePending,
    durable_write,
    operation_receipts,
    settle_operation,
)
from cannery.builds.sandbox import WorkerProfile
from cannery.builds.state import BuildResult, BuildState
from cannery.devices import Device
from cannery.git_intake import SnapshotBinding
from cannery.numerics.baseline_format import unpack_baseline
from cannery.numerics.baseline_state import BaselineState
from cannery.numerics.baselines import BaselinePending, BaselineUnavailable, cached_evidence
from cannery.numerics.executor import NumericalExecutor
from cannery.numerics.recipes import parse_recipes
from cannery.numerics.state import (
    NumericalClaim,
    NumericalResult,
    NumericalState,
    claim_numerical,
    current_numerical,
    finish_numerical,
    heartbeat,
    postpone_numerical,
)
from cannery.registry.freeze import BaselineEvidence
from cannery.registry.manifest import RecipeEntrypoint, validate_spec
from cannery.registry.models import SpecVersion
from cannery.submissions.models import Candidate, Submission


def discover(engine: Engine, *, device: Device = "a5") -> None:
    with Session(engine) as session, session.begin():
        rows = session.execute(
            select(Candidate.id, SpecVersion.smoke)
            .join(Submission, Submission.id == Candidate.id)
            .join(SpecVersion, SpecVersion.id == Submission.spec_id)
            .join(BuildState, BuildState.id == Candidate.id)
            .where(
                BuildState.status == "awaiting_validation",
                Candidate.manifest["implementation"]["device"].as_string() == device,
                ~select(NumericalState.id).where(NumericalState.id == Candidate.id).exists(),
            )
            .order_by(Candidate.id)
            .limit(50)
            .with_for_update(skip_locked=True)
        )
        for identifier, smoke in rows:
            try:
                baseline = BaselineEvidence.model_validate(smoke)
                row = session.get(BaselineState, baseline.baseline_key)
                if (
                    row is None
                    or row.status != "ready"
                    or row.artifact != baseline.artifact.model_dump(mode="json")
                ):
                    raise ValueError("baseline unavailable")
            except ValueError:
                session.add(
                    NumericalState(
                        id=identifier, status="blocked", error="recipe_baseline_required"
                    )
                )
                continue
            session.add(NumericalState(id=identifier, baseline_key=baseline.baseline_key))


class LeaseKeeper:
    def __init__(self, engine: Engine, claim: NumericalClaim) -> None:
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
            if self.lost.is_set() or current_numerical(session, self.claim) is None:
                raise RemotePending("Numerical ownership lost")

    def __enter__(self) -> "LeaseKeeper":
        self.thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop.set()
        self.thread.join(timeout=2)


def retained(engine: Engine, key: UUID, executor: BuildExecutor) -> dict[str, Any]:
    with Session(engine) as session:
        candidate = session.get(Candidate, key)
        request = session.get(Submission, key)
        build = session.get(BuildState, key)
        if (
            candidate is None
            or request is None
            or build is None
            or build.status != "awaiting_validation"
        ):
            raise ExecutionFailure("admitted_candidate_missing")
        result = session.get(BuildResult, build.attempt_id)
        frozen = session.get(SpecVersion, request.spec_id)
        if frozen is None or result is None or result.artifact is None:
            raise ExecutionFailure("admitted_candidate_missing")
        binding = SnapshotBinding.model_validate(frozen.snapshot)
        files = executor.snapshots.read(binding).files
        spec = validate_spec(files, binding.source.path)
        if spec.content_sha256 != frozen.content_sha256 or spec.file_sha256 != frozen.file_sha256:
            raise ExecutionFailure("retained_spec_mismatch")
        cases = spec.manifest.spec.cases
        if not isinstance(cases, RecipeEntrypoint):
            raise BaselineUnavailable("recipe_baseline_required")
        prefix = "" if binding.source.path == "." else binding.source.path + "/"
        suite = parse_recipes(files[prefix + cases.recipe_file])
        evidence = BaselineEvidence.model_validate(frozen.smoke)
        cache = session.get(BaselineState, evidence.baseline_key)
        if cache is None:
            raise BaselineUnavailable("baseline_missing")
        checked = cached_evidence(cache, spec, suite, executor.artifacts)
        if checked != evidence:
            raise BaselineUnavailable("baseline_identity_mismatch")
        package = executor.artifacts.get(checked.artifact)
        baseline = unpack_baseline(
            package,
            checked.artifact.sha256,
            spec_sha256=spec.content_sha256,
            suite=suite,
            environment=spec.environment,
        )
        return {
            "candidate": candidate,
            "spec": spec,
            "suite": suite,
            "baseline": baseline,
            "binary": ArtifactRef.model_validate(result.artifact),
            "baseline_artifact": checked.artifact,
        }


def prune(journal: Path) -> None:
    for name in ("validation.tar.gz", "results.json"):
        (journal / name).unlink(missing_ok=True)
    durable_write(journal / "cleanup-complete", b"complete\n")


def process(
    engine: Engine,
    key: UUID,
    factory: Callable[[NumericalClaim], NumericalExecutor],
    *,
    device: Device = "a5",
) -> None:
    claim = claim_numerical(engine, key, device=device)
    if claim is None:
        return
    with LeaseKeeper(engine, claim) as keeper:
        try:
            executor = factory(claim)
            executor.fence = keeper.fence
            failure_path = executor.remote.journal / "numerical-failure.json"
            try:
                if failure_path.exists():
                    stored = json.loads(failure_path.read_bytes())
                    if stored["blocked"]:
                        raise BaselineUnavailable(stored["error"])
                    raise ExecutionFailure(stored["error"], retryable=stored["retryable"])
                data = retained(engine, key, executor)
                report = executor.validate(attempt=claim.attempt, fence=keeper.fence, **data)
            except (ExecutionFailure, RemoteFailure, BaselineUnavailable) as failure:
                outcome: dict[str, Any] = {
                    "error": str(failure)
                    if isinstance(failure, BaselineUnavailable)
                    else failure.code
                    if isinstance(failure, ExecutionFailure)
                    else "worker_failed",
                    "retryable": failure.retryable
                    if isinstance(failure, ExecutionFailure)
                    else isinstance(failure, RemoteFailure),
                    "blocked": isinstance(failure, BaselineUnavailable),
                }
                keeper.fence()
                durable_write(failure_path, json.dumps(outcome, sort_keys=True).encode())
                if list(executor.remote.journal.glob("*.dispatch")):
                    executor.cleanup(claim.attempt)
                details = evidence(executor)
                keeper.fence()
                if finish_numerical(engine, claim, evidence=details, **outcome):
                    prune(executor.remote.journal)
                return
            executor.cleanup(claim.attempt)
            details = evidence(executor)
            keeper.fence()
            if finish_numerical(engine, claim, report=report, evidence=details):
                prune(executor.remote.journal)
        except (
            RemotePending,
            BaselinePending,
            ArtifactError,
            OSError,
            ValueError,
            SQLAlchemyError,
        ):
            logging.warning("Numerical operation pending: %s", key)
            postpone_numerical(engine, claim)


def cleanup_finished(
    engine: Engine, root: Path, factory: Callable[[NumericalClaim], NumericalExecutor]
) -> None:
    if not root.exists():
        return
    count = 0
    cursor = root / "cleanup-cursor"
    previous = cursor.read_text() if cursor.exists() else ""
    journals = sorted(root.iterdir(), key=lambda path: path.name)
    journals = [p for p in journals if p.name > previous] + [
        p for p in journals if p.name <= previous
    ]
    for journal in journals:
        if not journal.is_dir() or journal.is_symlink() or (journal / "cleanup-complete").exists():
            continue
        try:
            identifier = UUID(hex=journal.name)
        except ValueError:
            continue
        with Session(engine) as session:
            result = session.get(NumericalResult, identifier)
            if result is None:
                continue
        if count >= 5:
            break
        durable_write(cursor, journal.name.encode())
        count += 1
        try:
            executor = factory(NumericalClaim(result.candidate_id, result.id, result.id))
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
            logging.warning("Retaining numerical cleanup: %s", result.id)
