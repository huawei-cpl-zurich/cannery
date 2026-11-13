"""Host-only account provisioning. These commands require database access."""

import stat
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer
from sqlalchemy import Engine
from sqlalchemy.exc import SQLAlchemyError

from cannery.accounts import operator
from cannery.database import make_engine
from cannery.settings import Settings

app = typer.Typer(help="Operator-only local account management.", no_args_is_help=True)
PasswordFile = Annotated[Path | None, typer.Option(help="Private UTF-8 password file.")]


def secret(path: Path | None) -> str:
    if path is None:
        return str(typer.prompt("Password", hide_input=True, confirmation_prompt=True))
    # Open once and validate that same descriptor; never follow a symlink.
    import os

    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "r", encoding="utf-8") as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077 or info.st_uid != os.getuid():
            raise ValueError("Password file must be a user-owned private regular file")
        value = source.read(1026)
    return value.removesuffix("\n")


def execute(action: Callable[[Engine], None]) -> None:
    engine = make_engine(Settings())
    try:
        action(engine)
    except (ValueError, OSError, SQLAlchemyError):
        typer.echo("Account operation failed; verify input and database availability", err=True)
        raise typer.Exit(1) from None
    finally:
        engine.dispose()
    typer.echo("Account operation completed")


@app.command("create")
def create(username: str, password_file: PasswordFile = None, admin: bool = False) -> None:
    execute(
        lambda engine: operator.create(
            engine, username, secret(password_file), "admin" if admin else "user"
        )
    )


@app.command("reset-password")
def reset_password(username: str, password_file: PasswordFile = None) -> None:
    execute(lambda engine: operator.configure(engine, username, password=secret(password_file)))


@app.command("disable")
def disable(username: str) -> None:
    execute(lambda engine: operator.configure(engine, username, enabled=False))


@app.command("enable")
def enable(username: str) -> None:
    execute(lambda engine: operator.configure(engine, username, enabled=True))


@app.command("set-role")
def set_role(username: str, role: operator.Role) -> None:
    execute(lambda engine: operator.configure(engine, username, role=role))


@app.command("grant")
def grant(username: str, namespace: str) -> None:
    execute(lambda engine: operator.membership(engine, username, namespace, grant=True))


@app.command("revoke")
def revoke(username: str, namespace: str) -> None:
    execute(lambda engine: operator.membership(engine, username, namespace, grant=False))
