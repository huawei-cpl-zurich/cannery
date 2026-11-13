from datetime import UTC, datetime, timedelta

import pytest
from argon2.exceptions import VerifyMismatchError
from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from cannery.accounts import operator
from cannery.accounts.cli import secret
from cannery.accounts.models import Account, Credential, Membership
from cannery.cli import app
from cannery.database import Base


@pytest.fixture
def engine():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


def credential(engine, username="alice", namespace="team", kind="token"):
    with Session(engine) as db, db.begin():
        user = operator.account(db, username)
        db.add(
            Credential(
                account_id=user.id,
                namespace=namespace,
                kind=kind,
                digest=kind,
                expires_at=datetime.now(UTC) + timedelta(days=1),
            )
        )


def test_password_and_membership(engine):
    operator.create(engine, "alice", "a long password!")
    operator.membership(engine, "alice", "team", grant=True)
    operator.membership(engine, "alice", "team", grant=True)
    with Session(engine) as db:
        user = db.scalar(select(Account))
        assert user.enabled and user.role == "user"
        assert user.password_hash.startswith("$argon2id$")
        assert operator.PASSWORDS.verify(user.password_hash, "a long password!")
        with pytest.raises(VerifyMismatchError):
            operator.PASSWORDS.verify(user.password_hash, "wrong")
        assert len(list(db.scalars(select(Membership)))) == 1
    with pytest.raises(IntegrityError):
        operator.create(engine, "alice", "another password")


@pytest.mark.parametrize(
    "change",
    [{"password": "new long password"}, {"enabled": False}, {"enabled": True}, {"role": "admin"}],
)
def test_account_changes_revoke_all_credentials(engine, change):
    operator.create(engine, "alice", "a long password!")
    credential(engine)
    credential(engine, kind="session")
    operator.configure(engine, "alice", **change)
    with Session(engine) as db:
        assert list(db.scalars(select(Credential))) == []
        user = db.scalar(select(Account))
        for key, value in change.items():
            if key == "password":
                assert operator.PASSWORDS.verify(user.password_hash, value)
            else:
                assert getattr(user, key) == value


def test_membership_removal_is_scoped_and_idempotent(engine):
    operator.create(engine, "alice", "a long password!")
    operator.membership(engine, "alice", "team", grant=True)
    operator.membership(engine, "alice", "other", grant=True)
    credential(engine)
    credential(engine, namespace="other", kind="session")
    operator.membership(engine, "alice", "team", grant=False)
    operator.membership(engine, "alice", "team", grant=False)
    with Session(engine) as db:
        assert [row.namespace for row in db.scalars(select(Credential))] == ["other"]
        assert [row.namespace for row in db.scalars(select(Membership))] == ["other"]


def test_invalid_updates_do_not_revoke(engine):
    operator.create(engine, "alice", "a long password!")
    credential(engine)
    with pytest.raises(ValueError):
        operator.configure(engine, "alice", password="short")
    with pytest.raises(ValueError):
        operator.membership(engine, "alice", "invalid namespace", grant=True)
    with pytest.raises(ValueError, match="not found"):
        operator.configure(engine, "missing", enabled=False)
    with Session(engine) as db:
        assert len(list(db.scalars(select(Credential)))) == 1


def test_password_files_are_private_bounded_regular_files(tmp_path):
    path = tmp_path / "secret"
    path.write_text("a long password!\n")
    path.chmod(0o600)
    assert secret(path) == "a long password!"
    path.chmod(0o644)
    with pytest.raises(ValueError, match="private"):
        secret(path)
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(OSError):
        secret(link)
    with pytest.raises(ValueError):
        secret(tmp_path)
    path.chmod(0o600)
    path.write_text("x" * 2048)
    with pytest.raises(ValueError):
        operator.password_hash(secret(path))


def test_cli_does_not_echo_secrets_or_database_errors(monkeypatch, engine, tmp_path):
    monkeypatch.setattr("cannery.accounts.cli.make_engine", lambda settings: engine)
    path = tmp_path / "secret"
    path.write_text("a long password!")
    path.chmod(0o600)
    runner = CliRunner()
    result = runner.invoke(app, ["accounts", "create", "alice", "--password-file", str(path)])
    assert result.exit_code == 0, result.output
    assert "a long password!" not in result.output
    result = runner.invoke(app, ["accounts", "create", "alice", "--password-file", str(path)])
    assert result.exit_code == 1
    assert "INSERT" not in result.output and "argon2" not in result.output
