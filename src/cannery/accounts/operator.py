"""Account mutations serialize on the account row, including credential revocation."""

from typing import Literal

from argon2 import PasswordHasher
from pydantic import TypeAdapter
from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session

from cannery.accounts.models import Account, Credential, Membership
from cannery.registry.manifest import Name

PASSWORDS = PasswordHasher(time_cost=3, memory_cost=65536, parallelism=1)
NAME = TypeAdapter(Name)
Role = Literal["user", "admin"]


def password_hash(password: str) -> str:
    if not 12 <= len(password) <= 1024:
        raise ValueError("Password must contain 12–1024 characters")
    return PASSWORDS.hash(password)


def account(db: Session, username: str) -> Account:
    row = db.scalar(select(Account).where(Account.username == username).with_for_update())
    if row is None:
        raise ValueError("Account not found")
    return row


def create(engine: Engine, username: str, password: str, role: Role = "user") -> None:
    username = NAME.validate_python(username)
    if role not in ("user", "admin"):
        raise ValueError("Invalid role")
    digest = password_hash(password)
    with Session(engine) as db, db.begin():
        db.add(Account(username=username, password_hash=digest, role=role))


def configure(
    engine: Engine,
    username: str,
    *,
    password: str | None = None,
    enabled: bool | None = None,
    role: Role | None = None,
) -> None:
    digest = password_hash(password) if password is not None else None
    if role is not None and role not in ("user", "admin"):
        raise ValueError("Invalid role")
    with Session(engine) as db, db.begin():
        row = account(db, username)
        if digest is not None:
            row.password_hash = digest
        if enabled is not None:
            row.enabled = enabled
        if role is not None:
            row.role = role
        # Password resets, disable/re-enable and role changes require fresh login.
        db.execute(delete(Credential).where(Credential.account_id == row.id))


def membership(engine: Engine, username: str, namespace: str, *, grant: bool) -> None:
    namespace = NAME.validate_python(namespace)
    with Session(engine) as db, db.begin():
        row = account(db, username)
        existing = db.get(Membership, (row.id, namespace))
        if grant and existing is None:
            db.add(Membership(account_id=row.id, namespace=namespace))
        if not grant:
            if existing is not None:
                db.delete(existing)
            db.execute(
                delete(Credential).where(
                    Credential.account_id == row.id, Credential.namespace == namespace
                )
            )
