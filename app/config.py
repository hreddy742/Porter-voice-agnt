"""Application settings loaded from environment / `.env` via pydantic."""

from functools import lru_cache
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime config; DB_* and RECONTACT_* come from `.env`."""

    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "porter"
    db_user: str = "postgres"
    db_password: str = "postgres"

    recontact_default_days: int = 7

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def db_connect_kwargs(self) -> dict[str, Any]:
        """Keyword args for `psycopg2.connect`."""
        return {
            "host": self.db_host,
            "port": self.db_port,
            "database": self.db_name,
            "user": self.db_user,
            "password": self.db_password,
        }


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()


settings = get_settings()
