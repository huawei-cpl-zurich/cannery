import httpx
import typer

from cannery.accounts import client
from cannery.accounts.cli import PasswordFile, secret


def install(app: typer.Typer) -> None:
    @app.command("login")
    def login(
        url: str = "http://127.0.0.1:8000",
        username: str = "",
        namespace: str = "",
        password_file: PasswordFile = None,
    ) -> None:
        """Create a 30-day origin-bound CLI token using local account credentials."""
        try:
            base = client.secure(url)
            username = username or typer.prompt("Username")
            password = (
                secret(password_file)
                if password_file
                else typer.prompt("Password", hide_input=True)
            )
            with httpx.Client(timeout=5, follow_redirects=False, trust_env=False) as connection:
                reply = connection.post(
                    base + "/api/v1/auth/login",
                    headers={"X-Cannery-Login": "1"},
                    json={
                        "username": username,
                        "password": password,
                        "namespace": namespace or None,
                    },
                )
                reply.raise_for_status()
                csrf = {"X-CSRF-Token": reply.json()["csrf_token"]}
                try:
                    if not namespace:
                        current = connection.get(base + "/api/v1/auth/me")
                        current.raise_for_status()
                        namespace = current.json()["namespace"]
                    token = connection.post(
                        base + "/api/v1/auth/tokens",
                        headers=csrf,
                        json={"namespace": namespace, "label": "cannery CLI"},
                    )
                    token.raise_for_status()
                    client.save(base, token.json()["token"])
                finally:
                    connection.post(base + "/api/v1/auth/logout", headers=csrf)
            typer.echo("CLI credentials saved for this service (expires in 30 days)")
        except (httpx.HTTPError, ValueError, OSError, KeyError, TypeError):
            typer.echo(
                "Sign-in failed; verify credentials, namespace and service availability", err=True
            )
            raise typer.Exit(1) from None

    @app.command("logout")
    def logout(url: str = "http://127.0.0.1:8000") -> None:
        """Revoke this CLI token, then remove its local credentials."""
        try:
            headers = client.headers(url)
            if headers:
                reply = httpx.post(
                    client.secure(url) + "/api/v1/auth/logout", headers=headers, timeout=5
                )
                if reply.status_code != 401:
                    reply.raise_for_status()
            client.location(url).unlink(missing_ok=True)
            typer.echo("Signed out")
        except (httpx.HTTPError, ValueError, OSError):
            typer.echo("Logout unavailable; local credentials retained for retry", err=True)
            raise typer.Exit(1) from None
