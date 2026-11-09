import base64
import ipaddress
import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, SecretStr, field_validator

from cannery.git_intake import GitSource
from cannery.git_intake.process import IntakeError, clean_environment, git, run
from cannery.git_intake.tunnel import tunnel


class RemoteRule(BaseModel):
    """Operator configuration only; never accept this policy from request bodies."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    url: str
    namespaces: tuple[str, ...]
    allowed_networks: tuple[str, ...] = ()
    ca_file: Path | None = None
    username: str = "oauth2"
    token: SecretStr | None = None

    @field_validator("url")
    @classmethod
    def https_only(cls, value: str) -> str:
        source = GitSource(url=value, commit="0" * 40, path=".")
        if urlsplit(source.url).scheme != "https":
            raise ValueError("Only HTTPS Git transport is enabled")
        return source.url

    @field_validator("allowed_networks")
    @classmethod
    def networks(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        for value in values:
            ipaddress.ip_network(value)
        return values


def pinned_environment(rule: RemoteRule) -> dict[str, str]:
    parsed = urlsplit(rule.url)
    host = parsed.hostname
    port = parsed.port or 443
    # DNS lookup itself gets a process deadline; Git/curl then uses only the
    # checked address, so a second DNS answer cannot redirect a scoped token.
    raw = run(
        [
            sys.executable,
            "-c",
            "import socket,json,sys; print(json.dumps(sorted({r[4][0] for r in "
            "socket.getaddrinfo(sys.argv[1],int(sys.argv[2]),type=socket.SOCK_STREAM)})))",
            str(host),
            str(port),
        ],
        timeout=5,
    )
    addresses = json.loads(raw)
    if not addresses:
        raise IntakeError("Git remote has no permitted address")
    networks = [ipaddress.ip_network(value) for value in rule.allowed_networks]
    for value in addresses:
        address = ipaddress.ip_address(value)
        if not address.is_global and not any(address in network for network in networks):
            raise IntakeError("Git remote address is outside permitted networks")
    chosen = addresses[0]
    if ":" in chosen:
        chosen = f"[{chosen}]"
    config = {
        "http.curloptResolve": f"{host}:{port}:{chosen}",
        "http.followRedirects": "false",
        "http.proxy": "",
        "http.sslVerify": "true",
        "http.lowSpeedLimit": "1",
        "http.lowSpeedTime": "10",
        "credential.helper": "",
        "fetch.fsckObjects": "true",
        "transfer.unpackLimit": "1",
    }
    if rule.ca_file is not None:
        config["http.sslCAInfo"] = str(rule.ca_file)
    if rule.token is not None:
        raw_token = f"{rule.username}:{rule.token.get_secret_value()}".encode()
        config[f"http.{rule.url}.extraHeader"] = (
            "Authorization: Basic " + base64.b64encode(raw_token).decode()
        )
    env = clean_environment()
    env["CANNERY_PINNED_ADDRESS"] = addresses[0]
    env["GIT_CONFIG_COUNT"] = str(len(config))
    for i, (key, value) in enumerate(config.items()):
        env[f"GIT_CONFIG_KEY_{i}"] = key
        env[f"GIT_CONFIG_VALUE_{i}"] = value
    return env


@contextmanager
def fetched_source(
    source: GitSource, namespace: str, rules: tuple[RemoteRule, ...], scratch: Path | None = None
) -> Iterator[Path]:
    rule = next(
        (rule for rule in rules if rule.url == source.url and namespace in rule.namespaces), None
    )
    if rule is None:
        raise IntakeError("Git repository is not authorized for this namespace")
    env = pinned_environment(rule)
    parsed = urlsplit(rule.url)
    with tunnel(str(parsed.hostname), parsed.port or 443, env["CANNERY_PINNED_ADDRESS"]) as proxy:
        for i in range(int(env["GIT_CONFIG_COUNT"])):
            if env[f"GIT_CONFIG_KEY_{i}"] == "http.proxy":
                env[f"GIT_CONFIG_VALUE_{i}"] = proxy
        with TemporaryDirectory(prefix="cannery-intake-", dir=scratch) as temporary:
            repository = Path(temporary) / "source.git"
            git(
                repository,
                "init",
                "--bare",
                "--object-format=" + ("sha256" if len(source.commit) == 64 else "sha1"),
                str(repository),
            )
            git(
                repository,
                "fetch",
                "--depth=1",
                "--no-tags",
                "--no-recurse-submodules",
                source.url,
                f"{source.commit}:refs/intake/source",
                env=env,
            )
            if git(repository, "cat-file", "-t", source.commit).strip() != b"commit":
                raise IntakeError("Pinned source is not a commit")
            yield repository
