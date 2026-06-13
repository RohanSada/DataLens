"""Import an uploaded .sql file into a temporary SQLite database."""
from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path


def import_sql_file_to_sqlite(sql_bytes: bytes, filename: str) -> str:
    """Execute SQL script against a new SQLite DB and return the DB file path."""
    stem = Path(filename).stem or "imported_database"
    temp_dir = Path(tempfile.mkdtemp(prefix="datalens_sql_"))
    db_path = temp_dir / f"{stem}.sqlite"

    sql_text = sql_bytes.decode("utf-8", errors="replace").strip()
    if not sql_text:
        raise ValueError("The uploaded SQL file is empty.")

    connection = sqlite3.connect(db_path)
    try:
        connection.executescript(sql_text)
        connection.commit()
    except sqlite3.Error as exc:
        connection.close()
        db_path.unlink(missing_ok=True)
        raise ValueError(
            "Could not import SQL file into SQLite. "
            "The file must contain SQLite-compatible SQL (CREATE TABLE, INSERT, etc.). "
            f"Details: {exc}"
        ) from exc
    finally:
        connection.close()

    return str(db_path)
