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
            page.goto(base + "/account")
            page.locator("#account-identity").get_by_text("alice", exact=False).wait_for()
            page.locator('#account-token [name="label"]').fill("browser-token")
            page.get_by_role("button", name="Create 30-day CLI token").click()
            page.wait_for_function(
                "document.getElementById('account-new-token').textContent.length > 30"
            )
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


def assert_revoked(base, token):
    assert (
        httpx.get(
            base + "/api/v1/auth/me", headers={"Authorization": "Bearer " + token}
        ).status_code
        == 401
    )
