"""
Agni AI - Application Configuration

Provides the configuration required by the voice/STT services.

The existing project uses .env.local for local credentials, so this
configuration supports both .env and .env.local.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment files.

    Only the configuration needed by the current voice/STT integration
    is defined here. Additional application settings can be added later
    as the rest of the backend is integrated.
    """

    DEEPGRAM_API_KEY: str | None = None

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"),
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings object for the current process."""
    return Settings()


settings = get_settings()