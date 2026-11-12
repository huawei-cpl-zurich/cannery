"""Publish verified private blobs before atomically exposing a pinned release."""

import argparse
import json
import logging
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from uuid import UUID

from sqlalchemy import Engine
from sqlalchemy.exc import SQLAlchemyError

from cannery.artifacts import ArtifactError, ArtifactStore, digest
from cannery.artifacts.factory import make_store
from cannery.database import make_engine
from cannery.releases.assets import InvalidReleaseArtifact, verify_assets
from cannery.releases.contract import ReleaseManifest, read_manifest
from cannery.releases.eligibility import PublicationUnavailable, publication_manifest
from cannery.releases.state import Claim, claim_job, complete, due_jobs, heartbeat, retry_job
from cannery.settings import Settings


class OwnershipLost(Exception):
    pass


@contextmanager
def keep_lease(engine: Engine, claim: Claim) -> Iterator[Callable[[], None]]:
    stopped, lost = threading.Event(), threading.Event()

    def pulse() -> None:
        while not stopped.wait(15):
            try:
                if heartbeat(engine, claim):
                    continue
            except SQLAlchemyError:
                pass
            lost.set()
            return

    def fence() -> None:
        if lost.is_set() or not heartbeat(engine, claim):
            raise OwnershipLost

    thread = threading.Thread(target=pulse, daemon=True)
    thread.start()
    try:
        yield fence
    finally:
        stopped.set()
        thread.join(timeout=2)


def process(engine: Engine, identifier: UUID, store: ArtifactStore) -> None:
    claim = claim_job(engine, identifier)
    if claim is None:
        return
    with keep_lease(engine, claim) as fence:
        try:
            fence()
            manifest = publication_manifest(engine, claim.candidate_id)
            package = verify_assets(store, manifest, fence)
            fence()
            package_ref = store.put(f"releases/{claim.candidate_id}/package.tar", package)
            if package_ref.sha256 != manifest.package.sha256:
                raise InvalidReleaseArtifact("Package changed")
            data = manifest.model_dump(mode="json") | {
                "package": package_ref.model_dump(mode="json")
            }
            manifest = ReleaseManifest.model_validate_json(json.dumps(data))
            raw = manifest.encoded()
            fence()
            pin = store.put(f"releases/{claim.candidate_id}/manifest.json", raw)
            if pin.sha256 != digest(raw):
                raise InvalidReleaseArtifact("Manifest changed")
            fence()
            if read_manifest(store.get(pin), pin.sha256) != manifest:
                raise InvalidReleaseArtifact("Stored manifest changed")
            fence()
            complete(engine, claim, manifest, pin)
        except OwnershipLost:
            return
        except ArtifactError:
            retry_job(engine, claim, "release_artifact_unavailable")
        except PublicationUnavailable as error:
            retry_job(engine, claim, str(error), permanent=True)
        except ValueError:
            retry_job(engine, claim, "invalid_release_artifact", permanent=True)
        except SQLAlchemyError:
            # A lost commit reply resumes by the same durable job, never a new release.
            logging.warning("Publication control unavailable: %s", identifier)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    settings = Settings()
    if (
        settings.artifact_backend != "private"
        or not settings.artifact_read_token
        or not settings.artifact_write_token
    ):
        raise SystemExit("Configure private publication storage")
    engine = make_engine(settings)
    store = make_store(settings)
    stopped = threading.Event()
    try:
        while True:
            try:
                for identifier in due_jobs(engine):
                    process(engine, identifier, store)
            except (SQLAlchemyError, ArtifactError, OSError):
                logging.warning("Publication worker pending")
            if args.once:
                return
            stopped.wait(2)
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
