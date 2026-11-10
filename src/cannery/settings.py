from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from cannery.git_intake.transport import RemoteRule


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CANNERY_", extra="ignore")

    service_name: str = "cannery"

    database_url: SecretStr = SecretStr(
        "postgresql+psycopg://cannery:cannery@localhost:5432/cannery"
    )
    redis_url: SecretStr = SecretStr("redis://localhost:6379/0")

    dev_identity_enabled: bool = False
    dev_role: Literal["user", "admin"] = "user"

    artifact_backend: Literal["fake", "hf"] = "fake"
    artifact_root: Path = Path("/tmp/cannery-artifacts")
    hf_repository: str | None = None
    hf_token: SecretStr | None = None

    git_remotes: tuple[RemoteRule, ...] = ()
    snapshot_root: Path = Path("/var/lib/cannery/snapshots")
    intake_scratch: Path = Path("/tmp/cannery-intake")
    smoke_image: str = "cannery-reference:local"
    smoke_timeout: float = Field(default=30, gt=0, le=60)
