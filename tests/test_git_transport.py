import json
import sys

import pytest
from pydantic import SecretStr

from cannery.git_intake import GitSource
from cannery.git_intake.process import IntakeError, clean_environment, run
from cannery.git_intake.transport import RemoteRule, fetched_source, pinned_environment


def test_process_deadline_and_environment(monkeypatch) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "secret")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "100")
    assert "HTTPS_PROXY" not in clean_environment()
    assert "GIT_CONFIG_COUNT" not in clean_environment()
    with pytest.raises(IntakeError, match="deadline"):
        run([sys.executable, "-c", "import time;time.sleep(30)"], timeout=0.1)
    with pytest.raises(IntakeError, match="budget"):
        run([sys.executable, "-c", "print('x'*1000)"], max_output=100)


def test_scope_rejected_before_dns(monkeypatch) -> None:
    monkeypatch.setattr(
        "cannery.git_intake.transport.pinned_environment",
        lambda _: pytest.fail("must not resolve unauthorized repository"),
    )
    source = GitSource(url="https://example.com/private.git", commit="a" * 40, path="spec")
    rule = RemoteRule(url=source.url, namespaces=("other",))
    with pytest.raises(IntakeError, match="authorized"):
        with fetched_source(source, "team", (rule,)):
            pytest.fail("unauthorized fetch")


def test_address_pinning_and_token_scope(monkeypatch) -> None:
    monkeypatch.setattr("cannery.git_intake.transport.run", lambda *a, **k: b'["8.8.8.8"]')
    rule = RemoteRule(
        url="https://example.com/repo.git", namespaces=("team",), token=SecretStr("secret")
    )
    env = pinned_environment(rule)
    config = {
        env[f"GIT_CONFIG_KEY_{i}"]: env[f"GIT_CONFIG_VALUE_{i}"]
        for i in range(int(env["GIT_CONFIG_COUNT"]))
    }
    assert config["http.curloptResolve"] == "example.com:443:8.8.8.8"
    assert config["http.followRedirects"] == "false"
    assert config[f"http.{rule.url}.extraHeader"].startswith("Authorization: Basic ")
    assert "secret" not in repr(rule)


@pytest.mark.parametrize("addresses", [["127.0.0.1"], ["8.8.8.8", "169.254.169.254"], []])
def test_private_or_mixed_dns_rejected(monkeypatch, addresses) -> None:
    monkeypatch.setattr(
        "cannery.git_intake.transport.run", lambda *a, **k: json.dumps(addresses).encode()
    )
    with pytest.raises(IntakeError, match="address"):
        pinned_environment(RemoteRule(url="https://host/repo.git", namespaces=("team",)))


def test_failed_fetch_cleans_scratch(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "cannery.git_intake.transport.pinned_environment",
        lambda _: {"CANNERY_PINNED_ADDRESS": "127.0.0.1", "GIT_CONFIG_COUNT": "0"},
    )

    def failing(*args, **kwargs):
        raise IntakeError("injected network outage")

    monkeypatch.setattr("cannery.git_intake.transport.git", failing)
    source = GitSource(url="https://host/repo.git", commit="a" * 40, path="spec")
    with pytest.raises(IntakeError):
        with fetched_source(
            source, "team", (RemoteRule(url=source.url, namespaces=("team",)),), scratch=tmp_path
        ):
            pass
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("length,object_format", [(40, "sha1"), (64, "sha256")])
def test_fetch_initializes_matching_object_format(tmp_path, monkeypatch, length, object_format):
    from cannery.git_intake.process import git as real_git

    def simulated_fetch(repository, *args, **kwargs):
        if args[0] == "fetch":
            return b""
        if args[:2] == ("cat-file", "-t"):
            return b"commit\n"
        return real_git(repository, *args, **kwargs)

    monkeypatch.setattr("cannery.git_intake.transport.git", simulated_fetch)
    monkeypatch.setattr(
        "cannery.git_intake.transport.pinned_environment",
        lambda _: {"CANNERY_PINNED_ADDRESS": "127.0.0.1", "GIT_CONFIG_COUNT": "0"},
    )
    source = GitSource(url="https://host/repo.git", commit="a" * length, path="spec")
    with fetched_source(
        source, "team", (RemoteRule(url=source.url, namespaces=("team",)),)
    ) as repo:
        assert real_git(repo, "rev-parse", "--show-object-format").strip().decode() == object_format
