"""Registration handoff to the shared retained CPU oracle cache."""

from sqlalchemy import Engine
from sqlalchemy.orm import Session

from cannery.artifacts.factory import make_store
from cannery.database import aware
from cannery.git_intake import SnapshotBinding
from cannery.git_intake.snapshots import SnapshotStore
from cannery.numerics.baseline_format import baseline_key
from cannery.numerics.baseline_state import BaselineState, claim_baseline, ensure_baseline
from cannery.numerics.baselines import BaselinePending, BaselineUnavailable, advance_baseline
from cannery.numerics.oracle_job import OracleJob, OraclePending
from cannery.numerics.recipes import parse_recipes
from cannery.registry.freeze import BaselineEvidence
from cannery.registry.manifest import RecipeEntrypoint, ValidatedSpec
from cannery.registry.registrations import Claim, current_claim, retain_baseline_registration
from cannery.settings import Settings


def recipe_evidence(
    engine: Engine,
    claim: Claim,
    binding: SnapshotBinding,
    spec: ValidatedSpec,
    store: SnapshotStore,
    settings: Settings,
) -> BaselineEvidence:
    if settings.baseline_volume is None or settings.artifact_backend != "private":
        raise BaselineUnavailable("baseline_unconfigured")
    cases = spec.manifest.spec.cases
    if not isinstance(cases, RecipeEntrypoint):
        raise BaselineUnavailable("baseline_recipe_required")
    prefix = "" if binding.source.path == "." else binding.source.path + "/"
    suite = parse_recipes(store.read(binding).files[prefix + cases.recipe_file])
    key = baseline_key(spec.content_sha256, suite, spec.environment)
    if claim.prepared_snapshot is None:
        ensure_baseline(engine, key, spec.content_sha256, binding)
        if not retain_baseline_registration(engine, claim, binding, key):
            raise BaselinePending("registration_ownership_lost")
    elif claim.baseline_key != key:
        raise BaselineUnavailable("baseline_identity_mismatch")

    def fence() -> None:
        with Session(engine) as session:
            if current_claim(session, claim) is None:
                raise BaselinePending("registration_ownership_lost")

    with Session(engine) as session, session.begin():
        job = current_claim(session, claim)
        if job is None:
            raise BaselinePending("registration_ownership_lost")
        handle = "oracle:" + key
        if job.cleanup_handle not in (None, handle):
            raise ValueError("Previous sandbox cleanup is pending")
        job.cleanup_handle = handle
    return advance_baseline(
        engine,
        key,
        spec,
        suite,
        snapshots=store,
        artifacts=make_store(settings),
        root=settings.baseline_root,
        volume=settings.baseline_volume,
        image=settings.smoke_image,
        fence=fence,
    )


def cleanup_recipe(engine: Engine, handle: str, settings: Settings) -> bool:
    import re
    from datetime import UTC, datetime

    if re.fullmatch(r"oracle:[a-f0-9]{64}", handle) is None:
        raise ValueError("Invalid oracle cleanup handle")
    key = handle[7:]
    with Session(engine) as session:
        row = session.get(BaselineState, key)
        if row is None:
            return True
        active = row.status in ("queued", "generating")
        expired = aware(row.deadline) <= datetime.now(UTC)
    if active and expired:
        claim_baseline(engine, key)  # Deadline fences any previous coordinator.
    with Session(engine) as session:
        row = session.get(BaselineState, key)
        assert row is not None
        if row.status not in ("ready", "failed"):
            return False  # Another registration may still need this shared oracle.
        attempt = row.attempt
    if attempt is None:
        return True
    if settings.baseline_volume is None:
        raise OraclePending("Oracle volume unavailable for cleanup")
    job = OracleJob(
        settings.baseline_root,
        settings.baseline_volume,
        key,
        attempt,
        "sha256:" + "0" * 64,
        make_store(settings),
        lambda: None,
    )
    job.cleanup()
    return True
