"""Extract a BIRD-style schema record from a live SQLite database.

Output matches the structure used in ``dev_table.json`` / ``dev_tables.json``:
one object with ``db_id``, table/column lists, ``column_types``, ``primary_keys``,
and ``foreign_keys``.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any


def derive_db_id(db_path: str | None, explicit_db_id: str | None = None) -> str:
    if explicit_db_id and explicit_db_id.strip():
        return explicit_db_id.strip()
    if db_path:
        return Path(db_path).stem
    return "database"


def _normalize_sqlite_type(declared_type: str) -> str:
    """Map SQLite declared types to BIRD-style type strings."""
    normalized = (declared_type or "text").strip().upper()
    if re.search(r"\bINT\b", normalized):
        return "integer"
    if re.search(r"\b(REAL|FLOA|DOUB|NUMERIC|DECIMAL)\b", normalized):
        return "real"
    if re.search(r"\b(DATE|TIME)\b", normalized):
        return "date"
    if re.search(r"\bBOOL\b", normalized):
        return "integer"
    return "text"


def extract_bird_schema(connection: sqlite3.Connection, db_id: str) -> dict[str, Any]:
    """Introspect SQLite and return one BIRD ``dev_tables.json`` entry."""
    cursor = connection.cursor()

    cursor.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
        "ORDER BY name"
    )
    table_names_original = [row[0] for row in cursor.fetchall()]

    column_names_original: list[list[Any]] = [[-1, "*"]]
    column_names: list[list[Any]] = [[-1, "*"]]
    column_types: list[str] = []
    primary_keys: list[Any] = []
    foreign_keys: list[list[int]] = []

    # Map (table_name, column_name) -> global column index
    column_index: dict[tuple[str, str], int] = {}

    for table_idx, table_name in enumerate(table_names_original):
        cursor.execute(f'PRAGMA table_info("{table_name}")')
        columns = cursor.fetchall()

        table_pk_cols: list[int] = []

        for _cid, name, col_type, notnull, default_value, pk in columns:
            global_idx = len(column_names_original)
            column_index[(table_name, name)] = global_idx

            column_names_original.append([table_idx, name])
            column_names.append([table_idx, name])
            column_types.append(_normalize_sqlite_type(col_type))

            if pk:
                table_pk_cols.append(global_idx)

        if len(table_pk_cols) == 1:
            primary_keys.append(table_pk_cols[0])
        elif len(table_pk_cols) > 1:
            primary_keys.append(table_pk_cols)

        cursor.execute(f'PRAGMA foreign_key_list("{table_name}")')
        for _fk_id, _seq, ref_table, from_col, to_col, *_rest in cursor.fetchall():
            from_idx = column_index.get((table_name, from_col))
            to_idx = column_index.get((ref_table, to_col))
            if from_idx is not None and to_idx is not None:
                pair = [from_idx, to_idx]
                if pair not in foreign_keys:
                    foreign_keys.append(pair)

    return {
        "db_id": db_id,
        "table_names_original": table_names_original,
        "table_names": list(table_names_original),
        "column_names_original": column_names_original,
        "column_names": column_names,
        "column_types": column_types,
        "primary_keys": primary_keys,
        "foreign_keys": foreign_keys,
    }
