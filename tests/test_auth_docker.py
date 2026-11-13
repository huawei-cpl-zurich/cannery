"""Real PostgreSQL/Redis, browser/CLI and auth outage acceptance (CPU only)."""

import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("CANNERY_AUTH_DOCKER") != "1",
    reason="Set CANNERY_AUTH_DOCKER=1 for local Docker acceptance",
)
ROOT = Path(__file__).resolve().parents[1]


def docker(*args):
    return subprocess.check_output(["docker", *args], text=True, timeout=60).strip()


def until(action, timeout=30):
    end = time.monotonic() + timeout
    while True:
        try:
            return action()
        except (httpx.HTTPError, AssertionError, subprocess.CalledProcessError):
            if time.monotonic() >= end:
                raise
            time.sleep(0.25)


def free_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


@pytest.fixture
def deployment(tmp_path):
    slug = "cannery-m7c-" + uuid4().hex[:10]
    pg, redis = slug + "-pg", slug + "-redis"
    network = slug + "-network"
    process = None
    environment = os.environ.copy()
    environment.update(
        PYTHONPATH=str(ROOT / "src"),
        CANNERY_AUTH_ENABLED="true",
        CANNERY_DEV_IDENTITY_ENABLED="false",
        CANNERY_AUTH_COOKIE_SECURE="false",
        CANNERY_ARTIFACT_BACKEND="fake",
        CANNERY_ARTIFACT_ROOT=str(tmp_path / "artifacts"),
    )
    try:
        docker("network", "create", "--label", "cannery.task=m7c", network)
        docker(
            "run",
            "-d",
            "--name",
            pg,
            "--network",
            network,
            "--label",
            "cannery.task=m7c",
            "-e",
            "POSTGRES_PASSWORD=fixture",
            "-e",
            "POSTGRES_USER=cannery",
            "-e",
            "POSTGRES_DB=cannery",
            "-p",
            f"127.0.0.1:{free_port()}:5432",
            "postgres:17",
        )
        docker(
            "run",
            "-d",
            "--name",
            redis,
            "--network",
            network,
            "--label",
            "cannery.task=m7c",
            "-p",
            f"127.0.0.1:{free_port()}:6379",
            "redis:7-alpine",
        )
        pgport = docker("port", pg, "5432").rsplit(":", 1)[1]
        redisport = docker("port", redis, "6379").rsplit(":", 1)[1]
        environment["CANNERY_DATABASE_URL"] = (
            f"postgresql+psycopg://cannery:fixture@127.0.0.1:{pgport}/cannery"
        )
        environment["CANNERY_REDIS_URL"] = f"redis://127.0.0.1:{redisport}/0"
        until(lambda: docker("exec", pg, "pg_isready", "-U", "cannery"))
        subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", "head"],
            cwd=ROOT,
            env=environment,
            check=True,
            timeout=60,
            capture_output=True,
        )
        password = tmp_path / "password"
        password.write_text("test-only account password")
        password.chmod(0o600)
        for name, namespace in [("alice", "team"), ("bob", "other")]:
            for args in [
                ["create", name, "--password-file", str(password)],
                ["grant", name, namespace],
            ]:
                subprocess.run(
                    [sys.executable, "-c", "from cannery.cli import app; app()", "accounts", *args],
                    env=environment,
                    check=True,
                    timeout=15,
                    capture_output=True,
                )
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        base = f"http://127.0.0.1:{port}"
        environment["CANNERY_AUTH_ORIGIN"] = base
        with (tmp_path / "api.log").open("w") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "cannery.api:create_app",
                    "--factory",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                ],
                env=environment,
                stdout=log,
                stderr=log,
            )

            def ready():
                response = httpx.get(base + "/health/ready", timeout=3)
                response.raise_for_status()

            until(ready)
            yield base, environment, pg, redis, password
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        for name in [redis, pg]:
            subprocess.run(["docker", "rm", "-f", "-v", name], timeout=30, capture_output=True)
        subprocess.run(["docker", "network", "rm", network], timeout=30, capture_output=True)


def test_auth_browser_cli_and_outage_recovery(deployment, tmp_path):
    from playwright.sync_api import sync_playwright

    base, environment, pg, redis, password = deployment
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            executable_path=os.environ.get("CANNERY_BROWSER_EXECUTABLE") or None,
            args=["--no-sandbox"],
        )
        try:
            page = browser.new_page()
            page.goto(base + "/login")
            page.locator('[name="username"]').fill("alice")
            page.locator('[name="password"]').fill(password.read_text())
            page.get_by_role("button", name="Sign in", exact=True).click()
            page.wait_for_url(base + "/my-submissions")
            page.goto(base + "/register-kernel")
            page.get_by_label("Git repository URL").fill("https://example.com/kernel.git")
            page.get_by_label("Full commit SHA").fill("a" * 40)
            page.get_by_label("Source directory").fill("spec")
            page.get_by_role("button", name="Register specification", exact=True).click()
            page.wait_for_url("**/registrations/*")
            assert "queued" in page.locator("#registration").inner_text()
            page.goto(base + "/account")
            page.locator("#account-identity").get_by_text("alice", exact=False).wait_for()
            page.locator('#account-token [name="label"]').fill("browser-token")

            def disconnected(route):
                if route.request.method == "POST":
                    route.abort("internetdisconnected")
                else:
                    route.continue_()

            page.route("**/api/v1/auth/tokens", disconnected)
            page.get_by_role("button", name="Create 30-day CLI token").click()
            page.wait_for_function(
                "document.getElementById('account-error').textContent.length > 0"
            )
            assert page.locator("#account-new-token").inner_text() == ""
            page.unroute("**/api/v1/auth/tokens", disconnected)
            page.get_by_role("button", name="Create 30-day CLI token").click()
            page.wait_for_function(
                "document.getElementById('account-new-token').textContent.length > 30"
            )
            assert page.locator("#account-error").inner_text() == ""
            token = page.locator("#account-new-token").inner_text()
            assert (
                httpx.get(
                    base + "/api/v1/auth/me", headers={"Authorization": "Bearer " + token}
                ).status_code
                == 200
            )
            page.get_by_role("button", name="Revoke", exact=True).click()
            until(lambda: assert_revoked(base, token))
            page.get_by_role("button", name="Sign out", exact=True).click()
            page.wait_for_url(base + "/login")
        finally:
            browser.close()
    cli_env = environment | {"HOME": str(tmp_path / "cli-home")}
    command = [sys.executable, "-c", "from cannery.cli import app; app()"]
    subprocess.run(
        command + ["login", "--url", base, "--username", "alice", "--password-file", str(password)],
        env=cli_env,
        check=True,
        timeout=20,
        capture_output=True,
    )
    subprocess.run(
        command + ["submissions", "--url", base],
        env=cli_env,
        check=True,
        timeout=10,
        capture_output=True,
    )
    with httpx.Client(base_url=base, timeout=5) as connection:
        login = connection.post(
            "/api/v1/auth/login",
            headers={"X-Cannery-Login": "1"},
            json={"username": "alice", "password": password.read_text()},
        )
        assert login.status_code == 200
        csrf = {"X-CSRF-Token": login.json()["csrf_token"]}
        registered = connection.post(
            "/api/v1/kernels",
            headers=csrf,
            json={
                "source": {
                    "url": "https://example.com/kernel.git",
                    "commit": "a" * 40,
                    "path": "spec",
                }
            },
        )
        assert registered.status_code == 202
        with httpx.Client(base_url=base, timeout=5) as other:
            assert (
                other.post(
                    "/api/v1/auth/login",
                    headers={"X-Cannery-Login": "1"},
                    json={"username": "bob", "password": password.read_text()},
                ).status_code
                == 200
            )
            assert other.get(registered.json()["status_url"]).status_code == 404
        assert connection.get(registered.json()["status_url"]).status_code == 200
        docker("stop", "-t", "1", redis)
        assert connection.get("/api/v1/auth/me").status_code == 200
        failed = connection.post(
            "/api/v1/auth/login",
            headers={"X-Cannery-Login": "1"},
            json={"username": "alice", "password": password.read_text()},
        )
        assert failed.status_code == 503
        docker("start", redis)
        docker("stop", "-t", "1", pg)
        started = time.monotonic()
        assert connection.get("/api/v1/auth/me").status_code == 503
        assert time.monotonic() - started < 5
        assert connection.get("/health/live").status_code == 200
        assert connection.get("/static/style.css").status_code == 200
        assert connection.get("/api/v1/registrations").status_code == 503
        docker("start", pg)

        def recovered():
            assert connection.get("/api/v1/auth/me").status_code == 200

        until(recovered)
    subprocess.run(
        command + ["logout", "--url", base],
        env=cli_env,
        check=True,
        timeout=10,
        capture_output=True,
    )
    assert not list((Path(cli_env["HOME"]) / ".config/cannery/tokens").glob("*.json"))

    with httpx.Client(base_url=base, timeout=5) as connection:
        login = connection.post(
            "/api/v1/auth/login",
            headers={"X-Cannery-Login": "1"},
            json={"username": "alice", "password": password.read_text()},
        )
        csrf = {"X-CSRF-Token": login.json()["csrf_token"]}
        issued = connection.post("/api/v1/auth/tokens", headers=csrf, json={"namespace": "team"})
        token = issued.json()["token"]
        changed = connection.post(
            "/api/v1/auth/password",
            headers=csrf,
            json={
                "current_password": password.read_text(),
                "new_password": "new test-only account password",
            },
        )
        assert changed.status_code == 204
        assert connection.get("/api/v1/auth/me").status_code == 401
        assert_revoked(base, token)
        new_login = connection.post(
            "/api/v1/auth/login",
            headers={"X-Cannery-Login": "1"},
            json={"username": "alice", "password": "new test-only account password"},
        )
        assert new_login.status_code == 200
        assert (
            connection.post(
                "/api/v1/auth/logout", headers={"X-CSRF-Token": new_login.json()["csrf_token"]}
            ).status_code
            == 204
        )


def assert_revoked(base, token):
    assert (
        httpx.get(
            base + "/api/v1/auth/me", headers={"Authorization": "Bearer " + token}
        ).status_code
        == 401
    )


def test_postgres_revocation_between_credential_lookup_and_account_lock(deployment, monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from fastapi import HTTPException
    from sqlalchemy.orm import Session

    from cannery.accounts import auth, operator
    from cannery.accounts.models import Account
    from cannery.database import make_engine
    from cannery.settings import Settings

    base, environment, _, _, password = deployment
    with httpx.Client(base_url=base) as connection:
        reply = connection.post(
            "/api/v1/auth/login",
            headers={"X-Cannery-Login": "1"},
            json={"username": "alice", "password": password.read_text()},
        )
        assert reply.status_code == 200
        raw = connection.cookies["cannery_session"]
    engine = make_engine(Settings(database_url=environment["CANNERY_DATABASE_URL"]))
    reached, resume = threading.Event(), threading.Event()
    original = Session.get

    def get(db, entity, *args, **kwargs):
        if entity is Account:
            reached.set()
            assert resume.wait(5)
        return original(db, entity, *args, **kwargs)

    monkeypatch.setattr(Session, "get", get)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(auth.authenticate, engine, raw, "session")
            try:
                assert reached.wait(5)
                operator.configure(engine, "alice", password="a changed test-only password")
            finally:
                resume.set()
            with pytest.raises(HTTPException) as error:
                future.result(timeout=5)
            assert error.value.status_code == 401
    finally:
        engine.dispose()
