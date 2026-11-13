"""Opt-in trusted numerical worker; uses the registered A5 transport and policy."""

import argparse
import logging
import threading

from sqlalchemy.exc import SQLAlchemyError

from cannery.artifacts.factory import make_store
from cannery.builds.remote import CplRemote
from cannery.builds.sandbox import WorkerProfile
from cannery.database import make_engine
from cannery.git_intake.snapshots import SnapshotStore
from cannery.numerics.coordinator import cleanup_finished, discover, process
from cannery.numerics.executor import NumericalExecutor
from cannery.numerics.state import NumericalClaim, due_numericals
from cannery.settings import Settings
from cannery.shutdown import stop_on_term


def main() -> None:
    from uuid import UUID

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
        raise SystemExit("Configure private artifacts, worker profile and pinned loader first")
    profile = WorkerProfile.model_validate_json(settings.build_profile.read_bytes())
    store = make_store(settings)
    engine = make_engine(settings)
    wheel, checksum = settings.build_loader_wheel, settings.build_loader_sha256
    root = settings.build_control_root / "numerics"

    def factory(claim: NumericalClaim) -> NumericalExecutor:
        remote = CplRemote(
            settings.build_remote_client,
            root / claim.attempt.hex,
            profile.target,
            registry=settings.build_remote_registry,
        )
        return NumericalExecutor(
            profile, remote, SnapshotStore(settings.snapshot_root), store, wheel, checksum
        )

    stopped = threading.Event()
    restore_signals = stop_on_term(stopped)

    def clean() -> None:
        while not stopped.is_set():
            try:
                cleanup_finished(engine, root, factory, owner=profile)
            except (SQLAlchemyError, OSError):
                logging.warning("Numerical cleanup unavailable; retaining journals")
            stopped.wait(10)

    cleaner = threading.Thread(target=clean, daemon=True)
    if not args.once:
        cleaner.start()
    try:
        while not stopped.is_set():
            try:
                discover(engine, device=profile.device)
                for key in due_numericals(engine, device=profile.device):
                    if stopped.is_set():
                        break
                    process(engine, UUID(key), factory, device=profile.device)
            except (SQLAlchemyError, OSError, ValueError):
                logging.warning("Numerical worker unavailable; retrying")
            if args.once:
                cleanup_finished(engine, root, factory, owner=profile)
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
