"""Database-backed, revocable credentials. Raw secrets never enter the database."""

import hashlib
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID

from argon2.exceptions import VerificationError
from fastapi import HTTPException
from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session

from cannery.accounts.models import Account, Credential, Membership
from cannery.accounts.operator import PASSWORDS, account, password_hash
from cannery.database import aware

DUMMY_HASH = PASSWORDS.hash(secrets.token_urlsafe(32))


@dataclass(frozen=True)
class Principal:
    account_id: UUID
    username: str
    role: str
    namespace: str
    credential_id: UUID
    kind: str


def digest(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def csrf(raw: str) -> str:
    return digest("csrf:" + raw)


def namespaces(db: Session, identifier: UUID) -> list[str]:
    return list(
        db.scalars(
            select(Membership.namespace)
            .where(Membership.account_id == identifier)
            .order_by(Membership.namespace)
        )
    )


def permitted(db: Session, user: Account, namespace: str) -> bool:
    return bool(namespace) and (
        user.role == "admin" or db.get(Membership, (user.id, namespace)) is not None
    )


def issue(db: Session, user: Account, kind: str, namespace: str, label: str = "") -> str:
    if kind not in ("session", "token") or len(label) > 64:
        raise HTTPException(422, "Invalid credential")
    if not permitted(db, user, namespace):
        raise HTTPException(403, "Namespace access denied")
    raw = secrets.token_urlsafe(32)
    db.add(
        Credential(
            account_id=user.id,
            digest=digest(raw),
            kind=kind,
            namespace=namespace,
            label=label,
            expires_at=datetime.now(UTC)
            + (timedelta(hours=12) if kind == "session" else timedelta(days=30)),
        )
    )
    return raw


def login(engine: Engine, username: str, password: str, namespace: str | None) -> str:
    with Session(engine) as db, db.begin():
        user = db.scalar(select(Account).where(Account.username == username).with_for_update())
        try:
            PASSWORDS.verify(user.password_hash if user else DUMMY_HASH, password)
        except VerificationError:
            raise HTTPException(401, "Invalid credentials") from None
        if user is None or not user.enabled:
            raise HTTPException(401, "Invalid credentials")
        choices = namespaces(db, user.id)
        selected = namespace or (choices[0] if choices else "team" if user.role == "admin" else "")
        return issue(db, user, "session", selected)


def resolve(db: Session, raw: str, kind: str) -> Principal:
    credential = db.scalar(
        select(Credential).where(Credential.digest == digest(raw), Credential.kind == kind)
    )
    if credential is None or aware(credential.expires_at) <= datetime.now(UTC):
        raise HTTPException(401, "Authentication required")
    # Serialize with account changes and token issuance/revocation in mutations.
    user = db.get(Account, credential.account_id, with_for_update=True)
    if user is None or not user.enabled:
        raise HTTPException(401, "Authentication required")
    credential = db.scalar(
        select(Credential)
        .where(Credential.digest == digest(raw), Credential.kind == kind)
        .execution_options(populate_existing=True)
    )
    if credential is None or aware(credential.expires_at) <= datetime.now(UTC):
        raise HTTPException(401, "Authentication required")
    if not permitted(db, user, credential.namespace):
        raise HTTPException(403, "Namespace access denied")
    return Principal(
        user.id, user.username, user.role, credential.namespace, credential.id, credential.kind
    )


def authenticate(engine: Engine, raw: str, kind: str) -> Principal:
    with Session(engine) as db:
        return resolve(db, raw, kind)


def reset_own_password(engine: Engine, raw: str, kind: str, old: str, new: str) -> None:
    with Session(engine) as db, db.begin():
        principal = resolve(db, raw, kind)
        user = account(db, principal.username)
        try:
            PASSWORDS.verify(user.password_hash, old)
        except VerificationError:
            raise HTTPException(401, "Invalid credentials") from None
        user.password_hash = password_hash(new)
        db.execute(delete(Credential).where(Credential.account_id == user.id))
