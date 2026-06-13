"""Central configuration for the schema retrieval system.

All settings are loaded from a single ``settings.json`` file living next to this
module. Edit that file to change paths, database connections, embedding options
and retrieval tuning. Any key omitted from the JSON falls back to the in-code
default below.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

# schema_retrieval/ -> project root (DataLens/)
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
SETTINGS_FILE = PACKAGE_DIR / "settings.json"


class Settings(BaseModel):
    """Runtime configuration, loaded from ``settings.json``."""

    # --- Data source paths ---
    bird_root: Path = Field(
        default=PROJECT_ROOT / "Data" / "BIRD_SQL" / "minidev" / "MINIDEV"
    )
    # If unset, derived from bird_root.
    tables_json: Optional[Path] = None
    databases_dir: Optional[Path] = None

    # --- Qdrant (vector DB) ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    qdrant_collection: str = "bird_schema"

    # --- Neo4j (graph DB) ---
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password123"
    neo4j_database: str = "neo4j"

    # --- Embeddings ---
    embedding_provider: Literal["local", "openai"] = "local"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_batch_size: int = 64
    openai_api_key: Optional[str] = None
    openai_embedding_model: str = "text-embedding-3-small"

    # --- Value sampling ---
    sample_values: bool = True
    max_sample_values: int = 5
    # Cap rows scanned per column when collecting distinct values (large DBs).
    max_distinct_scan: int = 1000
    sample_only_text: bool = True
    max_value_char_len: int = 40

    # --- Retrieval tuning ---
    top_k_columns: int = 3
    top_k_tables: int = 3
    fk_expansion_hops: int = 1
    max_tables_in_context: int = 8
    # Approximate token budget for the assembled schema context.
    token_budget: int = 500

    # ------------------------------------------------------------------
    @field_validator("bird_root", "tables_json", "databases_dir", mode="after")
    @classmethod
    def _resolve_against_root(cls, value: Optional[Path]) -> Optional[Path]:
        """Resolve relative paths against the project root, not the cwd."""
        if value is None:
            return None
        return value if value.is_absolute() else (PROJECT_ROOT / value)

    # ------------------------------------------------------------------
    @property
    def tables_json_path(self) -> Path:
        return self.tables_json or (self.bird_root / "dev_tables.json")

    @property
    def databases_dir_path(self) -> Path:
        return self.databases_dir or (self.bird_root / "dev_databases")

    def sqlite_path(self, db_id: str) -> Path:
        return self.databases_dir_path / db_id / f"{db_id}.sqlite"

    def description_dir(self, db_id: str) -> Path:
        return self.databases_dir_path / db_id / "database_description"


def load_settings(path: str | Path) -> Settings:
    """Load settings from a specific JSON file (no caching)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Settings file not found at {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return Settings(**data)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance loaded from the default ``settings.json``."""
    data = {}
    if SETTINGS_FILE.exists():
        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    return Settings(**data)
