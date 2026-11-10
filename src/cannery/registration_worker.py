"""Privileged trusted coordinator; Docker socket is never given to author code."""

import json
import logging
import shutil
import sys
from pathlib import Path
from uuid import UUID

import dramatiq
from dramatiq.brokers.redis import RedisBroker
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from cannery.database import make_engine
from cannery.git_intake import SnapshotBinding
from cannery.git_intake.process import IntakeError, run
from cannery.git_intake.snapshots import SnapshotStore
from cannery.registry.freeze import SmokeEvidence, freeze_in_session
from cannery.registry.manifest import validate_spec
from cannery.registry.models import SpecConflict
from cannery.registry.registrations import (
    claim_registration,
    clear_cleanup,
    current_claim,
    fail_registration,
    pending_cleanup,
    prepare_handle,
)
from cannery.registry.sandbox import CleanupError, remove_smoke, split_handle
from cannery.settings import Settings

settings = Settings()
broker = RedisBroker(  # type: ignore[no-untyped-call]
    url=settings.redis_url.get_secret_value(),
    socket_connect_timeout=2,
    socket_timeout=2,
)
dramatiq.set_broker(broker)


def scratch_for(root: Path, handle: str) -> Path:
    _, owner = split_handle(handle)
    return root / f"attempt-{owner}"


def discard_scratch(root: Path, handle: str) -> None:
    path = scratch_for(root, handle)
    if path.exists():
        shutil.rmtree(path)


@dramatiq.actor(queue_name="registration", max_retries=0, time_limit=240000)
def register_spec(identifier: str) -> None:
    key = UUID(identifier)
    engine = make_engine(settings)
    claim = None
    try:
        leftover = pending_cleanup(engine, key)
        if leftover is not None:
            remove_smoke(leftover)
            discard_scratch(settings.intake_scratch, leftover)
            clear_cleanup(engine, key, leftover)
            return
        claim = claim_registration(engine, key)
        if claim is None:
            return
        if claim.previous_handle:
            remove_smoke(claim.previous_handle)
            discard_scratch(settings.intake_scratch, claim.previous_handle)
            clear_cleanup(engine, key, claim.previous_handle)
        handle = prepare_handle(engine, claim)
        settings.intake_scratch.mkdir(parents=True, exist_ok=True)
        scratch = scratch_for(settings.intake_scratch, handle)
        scratch.mkdir(mode=0o700)
        rules = []
        for rule in settings.git_remotes:
            values = rule.model_dump(mode="json")
            values["token"] = rule.token.get_secret_value() if rule.token is not None else None
            rules.append(values)
        payload = {
            "source": claim.source.model_dump(mode="json"),
            "namespace": claim.namespace,
            "name": claim.expected_name,
            "rules": rules,
            "scratch": str(scratch),
            "snapshots": str(settings.snapshot_root),
            "image": settings.smoke_image,
            "timeout": settings.smoke_timeout,
            "handle": handle,
        }
        try:
            raw = run(
                [sys.executable, "-m", "cannery.registry.admit"],
                data=json.dumps(payload).encode(),
                timeout=150,
            )
        finally:
            try:
                remove_smoke(handle)
                clear_cleanup(engine, key, handle)
            finally:
                discard_scratch(settings.intake_scratch, handle)
        result = json.loads(raw)
        if "error" in result:
            codes = {
                "invalid_spec",
                "cleanup_pending",
                "reference_failed",
                "intake_unavailable",
                "repository_denied",
            }
            code = result["error"] if result["error"] in codes else "worker_unavailable"
            fail_registration(engine, claim, code, bool(result.get("retryable", True)))
            return
        binding = SnapshotBinding.model_validate(result["binding"])
        evidence = SmokeEvidence.model_validate(result["evidence"])
        if binding.source != claim.source:
            raise SpecConflict("Source binding mismatch")
        store = SnapshotStore(settings.snapshot_root)
        spec = validate_spec(store.read(binding).files, binding.source.path)
        if claim.expected_name is not None and spec.manifest.kernel.name != claim.expected_name:
            raise SpecConflict("Kernel name mismatch")
        with Session(engine) as session, session.begin():
            job = current_claim(session, claim)
            if job is None:
                return
            frozen = freeze_in_session(
                session,
                spec,
                binding,
                evidence,
                store=store,
                namespace=claim.namespace,
                allow_new_version=claim.allow_new_version,
            )
            job.status, job.spec_id, job.error = "complete", frozen.id, None
            job.lease, job.lease_until = None, None
    except SpecConflict:
        if claim is not None:
            fail_registration(engine, claim, "spec_conflict", False)
    except (CleanupError, IntakeError, OSError, ValueError, SQLAlchemyError):
        logging.warning("Registration attempt unavailable: %s", key)
        if claim is not None:
            try:
                fail_registration(engine, claim, "worker_unavailable", True)
            except SQLAlchemyError:
                pass  # Durable lease expiry recovers a lost failure write.
    finally:
        engine.dispose()
