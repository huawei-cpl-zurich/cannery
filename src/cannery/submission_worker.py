"""Trusted Git-only intake; no author code or device execution at M2."""

import json
import logging
import shutil
import sys
from uuid import UUID

import dramatiq
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from cannery.database import make_engine
from cannery.git_intake import SnapshotBinding
from cannery.git_intake.process import IntakeError, run
from cannery.git_intake.snapshots import SnapshotStore
from cannery.registration_worker import broker, settings
from cannery.registry.models import SpecVersion
from cannery.submissions.freeze import freeze_candidate
from cannery.submissions.models import Submission, SubmissionConflict
from cannery.submissions.state import (
    claim_intake,
    clear_scratch,
    current_intake,
    fail_intake,
    pending_scratch,
    prepare_scratch,
)


def discard_scratch(key: UUID, token: UUID) -> None:
    path = settings.snapshot_root / ".submission-scratch" / f"{key.hex}-{token.hex}"
    if path.exists():
        shutil.rmtree(path)


@dramatiq.actor(broker=broker, queue_name="submission", max_retries=0, time_limit=240000)
def intake_submission(identifier: str) -> None:
    key = UUID(identifier)
    engine = make_engine(settings)
    claim = None
    try:
        previous = pending_scratch(engine, key)
        if previous is not None:
            discard_scratch(key, previous)
            clear_scratch(engine, key, previous)
            return
        claim = claim_intake(engine, key)
        if claim is None:
            return
        if claim.previous_scratch:
            discard_scratch(key, claim.previous_scratch)
            clear_scratch(engine, key, claim.previous_scratch)
        token = prepare_scratch(engine, claim)
        scratch = settings.snapshot_root / ".submission-scratch" / f"{key.hex}-{token.hex}"
        scratch.mkdir(mode=0o700, parents=True)
        with Session(engine) as session:
            request = session.get(Submission, key)
            assert request is not None
            spec = session.get(SpecVersion, request.spec_id)
            assert spec is not None
            kernel = spec.manifest["kernel"]
            payload = {
                "source": request.source,
                "namespace": request.namespace,
                "kernel": f"{kernel['namespace']}/{kernel['name']}",
                "spec_version": spec.version,
                "target": request.target,
                "scratch": str(scratch),
                "snapshots": str(settings.snapshot_root),
                "rules": [
                    rule.model_dump(mode="json")
                    | {"token": rule.token.get_secret_value() if rule.token else None}
                    for rule in settings.git_remotes
                ],
            }
        try:
            raw = run(
                [sys.executable, "-m", "cannery.submissions.admit"],
                data=json.dumps(payload).encode(),
                timeout=150,
            )
        finally:
            discard_scratch(key, token)
            clear_scratch(engine, key, token)
        result = json.loads(raw)
        if "error" in result:
            code = (
                result["error"]
                if result["error"]
                in {"repository_denied", "invalid_submission", "intake_unavailable"}
                else "worker_unavailable"
            )
            fail_intake(engine, claim, code, bool(result.get("retryable", True)))
            return
        snapshot = SnapshotBinding.model_validate(result["snapshot"])
        with Session(engine) as session, session.begin():
            state = current_intake(session, claim)
            if state is None:
                return
            freeze_candidate(session, key, snapshot, store=SnapshotStore(settings.snapshot_root))
            state.status, state.error = "awaiting_build", None
            state.lease, state.lease_until = None, None
    except SubmissionConflict:
        if claim is not None:
            fail_intake(engine, claim, "binding_mismatch", False)
    except (IntakeError, OSError, ValueError, SQLAlchemyError):
        logging.warning("Submission intake unavailable: %s", key)
        if claim is not None:
            try:
                fail_intake(engine, claim, "worker_unavailable", True)
            except SQLAlchemyError:
                pass
    finally:
        engine.dispose()
