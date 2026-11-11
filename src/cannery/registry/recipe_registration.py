"""Registration handoff to the shared retained CPU oracle cache."""

from sqlalchemy import Engine
from sqlalchemy.orm import Session

from cannery.artifacts.factory import make_store
from cannery.git_intake import SnapshotBinding
from cannery.git_intake.snapshots import SnapshotStore
from cannery.numerics.baseline_format import baseline_key
from cannery.numerics.baseline_state import ensure_baseline
from cannery.numerics.baselines import BaselinePending, BaselineUnavailable, advance_baseline
from cannery.numerics.oracle_job import OraclePending
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

    from cannery.numerics.baselines import cleanup_baseline

    if re.fullmatch(r"oracle:[a-f0-9]{64}", handle) is None:
        raise ValueError("Invalid oracle cleanup handle")
    if settings.baseline_volume is None:
        raise OraclePending("Oracle volume unavailable for cleanup")
    return cleanup_baseline(
        engine,
        handle[7:],
        root=settings.baseline_root,
        volume=settings.baseline_volume,
        artifacts=make_store(settings),
        fence=lambda: None,
    )
