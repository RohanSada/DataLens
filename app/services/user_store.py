"""SQLite-backed user and API key store."""

from __future__ import annotations

import secrets
import sqlite3
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.core.config import settings


@dataclass
class User:
    user_id: str
    email: str
    tenant_id: str
    password_hash: str
    api_key: str


class UserStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    tenant_id TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    api_key TEXT UNIQUE NOT NULL
                )
                """
            )
            conn.commit()

    def create_user(
        self,
        email: str,
        password_hash: str,
        tenant_id: str | None = None,
    ) -> User:
        user_id = str(uuid.uuid4())
        tenant = tenant_id or str(uuid.uuid4())
        api_key = f"dl_{secrets.token_urlsafe(32)}"
        user = User(
            user_id=user_id,
            email=email.lower(),
            tenant_id=tenant,
            password_hash=password_hash,
            api_key=api_key,
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO users (user_id, email, tenant_id, password_hash, api_key)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user.user_id, user.email, user.tenant_id, user.password_hash, user.api_key),
            )
            conn.commit()
        return user

    def get_user(self, user_id: str) -> User | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE user_id = ?", (user_id,)
            ).fetchone()
        return self._row_to_user(row) if row else None

    def get_user_by_email(self, email: str) -> User | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE email = ?", (email.lower(),)
            ).fetchone()
        return self._row_to_user(row) if row else None

    def get_user_by_api_key(self, api_key: str) -> User | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE api_key = ?", (api_key,)
            ).fetchone()
        return self._row_to_user(row) if row else None

    @staticmethod
    def _row_to_user(row: sqlite3.Row) -> User:
        return User(
            user_id=row["user_id"],
            email=row["email"],
            tenant_id=row["tenant_id"],
            password_hash=row["password_hash"],
            api_key=row["api_key"],
        )


@lru_cache
def get_user_store() -> UserStore:
    return UserStore(settings.user_store_path)
