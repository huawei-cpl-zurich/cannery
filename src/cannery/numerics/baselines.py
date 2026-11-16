"""Ingress-driven shared CPU output cache: ready baselines never execute again."""

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from sqlalchemy import Engine
from sqlalchemy.orm import Session

from cannery.artifacts import ArtifactError, ArtifactRef, ArtifactStore, digest
from cannery.builds.remote import durable_write
from cannery.database import aware
from cannery.git_intake import SnapshotBinding
from cannery.git_intake.process import IntakeError
from cannery.git_intake.snapshots import SnapshotStore
from cannery.numerics.baseline_format import BaselineBundle, baseline_key, unpack_baseline
from cannery.numerics.baseline_state import (
    BaselineState,
    claim_baseline,
    current_baseline,
    finish_baseline,
    postpone_baseline,
)
from cannery.numerics.baseline_v2 import BaselineBundleV2, baseline_key_v2, unpack_baseline_v2
from cannery.numerics.oracle_job import OracleFailure, OracleJob, OraclePending, docker
from cannery.numerics.recipes import RecipeSuite, parse_recipes
from cannery.numerics.recipes_v2 import RecipeSuiteV2
from cannery.numerics.tensor_contract import TensorContractV2
from cannery.registry.freeze import BaselineEvidence
from cannery.registry.manifest import RecipeEntrypoint, ValidatedSpec, validate_spec
from cannery.registry.tensor_manifest import validate_tensor_spec


class BaselinePending(Exception):
    pass


class BaselineUnavailable(ValueError):
    pass


def _expected_key(
    spec: ValidatedSpec,
    suite: RecipeSuite | RecipeSuiteV2,
    contract: TensorContractV2 | None,
) -> str:
    if isinstance(suite, RecipeSuiteV2):
        if spec.manifest.schema_version != 2 or contract is None:
            raise BaselineUnavailable("baseline_tensor_contract_missing")
        contract.validate_suite(suite)
        return baseline_key_v2(
            spec.content_sha256, suite, spec.environment, contract.output_contract()
        )
    if spec.manifest.schema_version != 1 or contract is not None:
        raise BaselineUnavailable("baseline_recipe_version_mismatch")
    return baseline_key(spec.content_sha256, suite, spec.environment)


def cached_evidence(
    row: BaselineState,
    spec: ValidatedSpec,
    suite: RecipeSuite | RecipeSuiteV2,
    artifacts: ArtifactStore,
    *,
    contract: TensorContractV2 | None = None,
) -> BaselineEvidence:
    if row.key != _expected_key(spec, suite, contract):
        raise BaselineUnavailable("baseline_identity_mismatch")
    if row.status != "ready" or row.artifact is None:
        raise BaselinePending("baseline_not_ready")
    try:
        reference = ArtifactRef.model_validate(row.artifact)
        package = artifacts.get(reference)
    except ArtifactError:
        raise BaselinePending("baseline_storage_unavailable") from None
    try:
        if digest(package) != reference.sha256:
            raise ValueError("digest mismatch")
        bundle: BaselineBundle | BaselineBundleV2
        if isinstance(suite, RecipeSuiteV2):
            assert contract is not None
            bundle = unpack_baseline_v2(
                package,
                reference.sha256,
                spec_sha256=spec.content_sha256,
                suite=suite,
                environment=spec.environment,
                contract=contract.output_contract(),
            )
        else:
            bundle = unpack_baseline(
                package,
                reference.sha256,
                spec_sha256=spec.content_sha256,
                suite=suite,
                environment=spec.environment,
            )
        return BaselineEvidence(
            spec_sha256=spec.content_sha256,
            baseline_key=row.key,
            suite_sha256=suite.sha256,
            generator=suite.generator,
            image_id=bundle.metadata["image_id"],
            python=spec.environment.python,
            torch=spec.environment.torch,
            cases=len(suite.cases),
            artifact=reference,
        )
    except ValueError:
        raise BaselineUnavailable("baseline_cache_invalid") from None


def advance_baseline(
    engine: Engine,
    key: str,
    spec: ValidatedSpec,
    suite: RecipeSuite | RecipeSuiteV2,
    *,
    contract: TensorContractV2 | None = None,
    snapshots: SnapshotStore,
    artifacts: ArtifactStore,
    root: Path,
    volume: str,
    image: str,
    fence: Callable[[], None],
) -> BaselineEvidence:
    if key != _expected_key(spec, suite, contract):
        raise BaselineUnavailable("baseline_identity_mismatch")

    def terminal_cleanup() -> None:
        try:
            cleanup_baseline(
                engine, key, root=root, volume=volume, artifacts=artifacts, fence=fence
            )
        except (OraclePending, ArtifactError, OSError):
            raise BaselinePending("baseline_cleanup_pending") from None

    fence()
    with Session(engine) as session:
        row = session.get(BaselineState, key)
        if row is None or row.spec_sha256 != spec.content_sha256:
            raise BaselineUnavailable("baseline_missing")
        if row.status == "ready":
            return cached_evidence(row, spec, suite, artifacts, contract=contract)
        if row.status == "failed":
            terminal_cleanup()
            raise BaselineUnavailable(row.error or "baseline_failed")
        binding = SnapshotBinding.model_validate(row.snapshot)
    claim = claim_baseline(engine, key)
    if claim is None:
        with Session(engine) as session:
            latest = session.get(BaselineState, key)
            if latest is not None and latest.status == "failed":
                terminal_cleanup()
                raise BaselineUnavailable(latest.error or "baseline_failed")
        raise BaselinePending("baseline_busy")

    def owned() -> None:
        fence()
        with Session(engine) as session:
            if current_baseline(session, claim) is None:
                raise OraclePending("Baseline ownership lost")

    job = None
    try:
        journal = root / key / claim.attempt.hex
        journal.mkdir(mode=0o700, parents=True, exist_ok=True)
        pinned = journal / "oracle-image"
        if not pinned.exists():
            code, output = docker(["image", "inspect", "--format", "{{.Id}}", "--", image])
            if code:
                raise OraclePending("Oracle image unavailable")
            owned()
            durable_write(pinned, output.strip())
        job = OracleJob(root, volume, key, claim.attempt, pinned.read_text(), artifacts, owned)
        files = snapshots.read(binding).files
        canonical_suite: RecipeSuite | RecipeSuiteV2
        if isinstance(suite, RecipeSuiteV2):
            frozen = validate_tensor_spec(files, binding.source.path)
            canonical, canonical_suite = frozen.spec, frozen.suite
            if frozen.contract != contract:
                raise OracleFailure("Retained tensor contract mismatch")
        else:
            canonical = validate_spec(files, binding.source.path)
            cases = canonical.manifest.spec.cases
            if not isinstance(cases, RecipeEntrypoint):
                raise OracleFailure("Oracle source must declare recipes")
            prefix = "" if binding.source.path == "." else binding.source.path + "/"
            canonical_suite = parse_recipes(files[prefix + cases.recipe_file])
        if canonical.content_sha256 != spec.content_sha256 or canonical_suite != suite:
            raise OracleFailure("Retained baseline source mismatch")
        reference = job.progress(canonical, canonical_suite, files, binding.source.path)
        owned()
        if not finish_baseline(engine, claim, artifact=reference):
            raise OraclePending("Baseline completion ownership lost")
    except (OraclePending, ArtifactError, OSError, IntakeError):
        postpone_baseline(engine, claim)
        raise BaselinePending("baseline_operation_pending") from None
    except ValueError:
        if job is not None:
            try:
                job.cleanup()
            except (OraclePending, OSError):
                postpone_baseline(engine, claim)
                raise BaselinePending("baseline_cleanup_pending") from None
            except OracleFailure:
                pass  # Never remove a container owned by somebody else.
        finish_baseline(engine, claim, error="reference_failed")
        raise BaselineUnavailable("reference_failed") from None
    with Session(engine) as session:
        complete = session.get(BaselineState, key)
        assert complete is not None
        return cached_evidence(complete, spec, suite, artifacts, contract=contract)


def cleanup_baseline(
    engine: Engine,
    key: str,
    *,
    root: Path,
    volume: str,
    artifacts: ArtifactStore,
    fence: Callable[[], None],
) -> bool:
    """Remove only terminal owned jobs; shared active baselines must survive."""
    with Session(engine) as session:
        row = session.get(BaselineState, key)
        if row is None:
            return True
        expired = aware(row.deadline) <= datetime.now(UTC)
        active = row.status in ("queued", "generating")
    if active and expired:
        claim_baseline(engine, key)
    with Session(engine) as session:
        row = session.get(BaselineState, key)
        assert row is not None
        if row.status not in ("ready", "failed"):
            return False
        attempt = row.attempt
    if attempt is None:
        return True
    assert isinstance(attempt, UUID)
    job = OracleJob(root, volume, key, attempt, "sha256:" + "0" * 64, artifacts, fence)
    job.cleanup()
    for name in ("input/plan.json", "output/result.json"):
        (root / key / attempt.hex / name).unlink(missing_ok=True)
    return True
