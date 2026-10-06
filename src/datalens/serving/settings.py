"""API settings, read from ``DATALENS_*`` environment variables."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DATALENS_", env_file=".env", extra="ignore")

    backend: Literal["openai", "anthropic"] = "openai"
    model: str = "datalens"
    base_url: str | None = "http://localhost:8001/v1"
    api_key: str | None = None
    effort: str | None = None

    db_dir: Path = Path("data/databases")
    samples: int = 5
    temperature: float | None = None  # default: 0.8, or none for Claude (it rejects the parameter)
    max_tokens: int | None = None  # default: 1,024, or 16,000 for Claude (thinking counts too)
    reasoning: bool = True
    num_examples: int = 3
    schema_cache_dir: str | None = ".cache/schemas"

    max_rows: int = 500
    query_timeout_s: float = 15.0
    cors_origins: list[str] = []
