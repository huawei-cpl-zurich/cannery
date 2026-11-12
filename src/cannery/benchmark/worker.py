"""Opt-in benchmark worker for existing and future numerically passed binaries."""

import argparse
import logging
import threading
import time

from sqlalchemy.exc import SQLAlchemyError

from cannery.artifacts import ArtifactError
from cannery.artifacts.factory import make_store
from cannery.benchmark.coordinator import cleanup_finished, process
from cannery.benchmark.device import device_cohort
from cannery.benchmark.executor import BenchmarkExecutor
from cannery.benchmark.scheduling import discover
from cannery.benchmark.state import PerformanceClaim, due_jobs
from cannery.builds.remote import CplRemote, RemoteFailure, RemotePending
from cannery.builds.sandbox import WorkerProfile
from cannery.database import make_engine
from cannery.git_intake.snapshots import SnapshotStore
from cannery.numerics.baselines import BaselinePending, BaselineUnavailable
from cannery.settings import Settings


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
        raise SystemExit("Configure private artifacts, worker profile and pinned loader first")
    profile = WorkerProfile.model_validate_json(settings.build_profile.read_bytes())
    store = make_store(settings)
    engine = make_engine(settings)
    root = settings.build_control_root / "benchmarks"
    wheel, checksum = settings.build_loader_wheel, settings.build_loader_sha256

    def factory(claim: PerformanceClaim) -> BenchmarkExecutor:
        remote = CplRemote(
            settings.build_remote_client,
            root / claim.attempt.hex,
            profile.target,
            registry=settings.build_remote_registry,
        )
        return BenchmarkExecutor(
            profile,
            remote,
            SnapshotStore(settings.snapshot_root),
            store,
            wheel,
            checksum,
        )

    stopped = threading.Event()

    def clean() -> None:
        while not stopped.is_set():
            try:
                cleanup_finished(engine, root, factory)
            except (SQLAlchemyError, OSError):
                logging.warning("Performance cleanup pending")
            stopped.wait(10)

    cleaner = threading.Thread(target=clean, daemon=True)
    if not args.once:
        cleaner.start()
    try:
        while True:
            try:
                cohort = device_cohort(
                    profile,
                    settings.build_control_root / "benchmark-identity",
                    settings.build_remote_client,
                    settings.build_remote_registry,
                    settings.build_loader_sha256,
                )
                discover(engine, cohort)
            except (
                SQLAlchemyError,
                OSError,
                ValueError,
                RemotePending,
                RemoteFailure,
                ArtifactError,
                BaselinePending,
                BaselineUnavailable,
            ):
                logging.warning("Performance worker pending; retaining attempts")
            try:
                for key in due_jobs(engine):
                    process(engine, key, factory)
            except (SQLAlchemyError, OSError, ValueError):
                logging.warning("Performance coordinator pending; retaining attempts")
            if args.once:
                cleanup_finished(engine, root, factory)
                return
            time.sleep(5)
    finally:
        stopped.set()
        if cleaner.is_alive():
            cleaner.join(timeout=2)
        engine.dispose()


if __name__ == "__main__":
    main()
