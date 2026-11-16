"""Immutable, content-pinned virtual Hub snapshots; no artifact copies."""

import hashlib
import json
from uuid import UUID

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from cannery.artifacts import ArtifactRef, digest
from cannery.releases import catalog
from cannery.releases.models import HubSnapshot, Release


def descriptor(release: Release) -> dict[str, str | int]:
    manifest = catalog.manifest_of(release)
    pin = ArtifactRef.model_validate(release.manifest)
    if pin.sha256 != digest(manifest.encoded()):
        raise ValueError("Release manifest pin differs")
    return {
        "schema": 1,
        "release_id": str(release.id),
        "repo_id": manifest.kernel,
        "manifest_sha256": pin.sha256,
        "package_sha256": manifest.package.sha256,
    }


def revision(document: dict[str, str | int]) -> str:
    raw = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    # Hub clients require a 40-hex commit identity. Integrity uses the SHA256 pins.
    return hashlib.sha1(b"cannery-hub-v1\0" + raw, usedforsecurity=False).hexdigest()


def ensure_snapshot(db: Session, release: Release) -> HubSnapshot:
    db.execute(select(Release.id).where(Release.id == release.id).with_for_update()).one()
    document = descriptor(release)
    expected = revision(document)
    row = db.get(HubSnapshot, release.id)
    if row is None:
        row = HubSnapshot(release_id=release.id, revision=expected, descriptor=document)
        db.add(row)
        db.flush()
    elif row.revision != expected or row.descriptor != document:
        raise ValueError("Immutable Hub snapshot differs")
    return row


def backfill(engine: Engine) -> int:
    """Page through published releases, committing each mapping atomically."""
    count = 0
    cursor: UUID | None = None
    while True:
        with Session(engine) as db:
            query = catalog.published().order_by(Release.id).limit(100)
            if cursor is not None:
                query = query.where(Release.id > cursor)
            identifiers = [r.id for r, _ in db.execute(query)]
        if not identifiers:
            return count
        for identifier in identifiers:
            with Session(engine) as db, db.begin():
                release, _ = catalog.release_info(db, identifier)
                created = db.get(HubSnapshot, identifier) is None
                ensure_snapshot(db, release)
                count += int(created)
        cursor = identifiers[-1]
