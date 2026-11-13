import ipaddress
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from cannery.git_intake.transport import RemoteRule
from cannery.registry.manifest import Name


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CANNERY_", extra="ignore", hide_input_in_errors=True
    )

    @model_validator(mode="after")
    def authentication_configuration(self) -> "Settings":
        if not self.auth_enabled:
            return self
        if self.dev_identity_enabled:
            raise ValueError("Account authentication and development identity cannot be combined")
        parsed = urlsplit(self.auth_origin)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or any((parsed.username, parsed.password, parsed.query, parsed.fragment, parsed.path))
        ):
            raise ValueError("Authentication origin must be an HTTP origin")
        try:
            loopback = ipaddress.ip_address(parsed.hostname).is_loopback
        except ValueError:
            loopback = parsed.hostname == "localhost"
        if not loopback and (parsed.scheme != "https" or not self.auth_cookie_secure):
            raise ValueError("Public authentication requires HTTPS and Secure cookies")
        return self

    service_name: str = "cannery"
    release_enabled: bool = False

    database_url: SecretStr = SecretStr(
        "postgresql+psycopg://cannery:cannery@localhost:5432/cannery"
    )
    redis_url: SecretStr = SecretStr("redis://localhost:6379/0")

    auth_enabled: bool = False
    auth_origin: str = "http://127.0.0.1:8095"
    auth_cookie_secure: bool = True

    dev_identity_enabled: bool = False
    dev_role: Literal["user", "admin"] = "user"

    artifact_backend: Literal["fake", "hf", "private"] = "fake"
    artifact_root: Path = Path("/tmp/cannery-artifacts")
    artifact_url: str = "http://artifacts:8000"
    artifact_read_token: SecretStr | None = None
    artifact_write_token: SecretStr | None = None
    artifact_previous_read_token: SecretStr | None = None
    artifact_previous_write_token: SecretStr | None = None
    hf_repository: str | None = None
    hf_token: SecretStr | None = None

    dev_namespace: Name = "team"
    git_remotes: tuple[RemoteRule, ...] = ()
    snapshot_root: Path = Path("/var/lib/cannery/snapshots")
    intake_scratch: Path = Path("/tmp/cannery-intake")
    smoke_image: str = "cannery-reference:local"
    smoke_timeout: float = Field(default=30, gt=0, le=60)

    benchmark_enabled: bool = False

    baseline_volume: str | None = None
    baseline_root: Path = Path("/var/lib/cannery/baselines")

    build_profile: Path | None = None
    build_loader_wheel: Path | None = None
    build_loader_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    build_control_root: Path = Path("/var/lib/cannery/build-control")
    build_remote_client: Path = Path.home() / ".agents/skills/remote-access/scripts/cpl-remote"
    build_remote_registry: Path = Path.home() / ".config/cpl/remotes.yaml"
