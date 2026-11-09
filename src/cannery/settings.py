from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CANNERY_", extra="ignore")

    service_name: str = "cannery"

    database_url: SecretStr = SecretStr(
        "postgresql+psycopg://cannery:cannery@localhost:5432/cannery"
    )
    redis_url: SecretStr = SecretStr("redis://localhost:6379/0")

    dev_identity_enabled: bool = False
    dev_role: Literal["user", "admin"] = "user"
