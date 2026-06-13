"""Audit log for query execution."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import Settings


class AuditLogger:
    def __init__(self, settings: Settings):
        self.db_path = Path(settings.user_store_path).parent / "audit.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    details TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def log(
        self,
        *,
        user_id: str,
        tenant_id: str,
        session_id: str,
        action: str,
        details: dict,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO audit_events (timestamp, user_id, tenant_id, session_id, action, details)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    user_id,
                    tenant_id,
                    session_id,
                    action,
                    json.dumps(details),
                ),
            )
            conn.commit()
