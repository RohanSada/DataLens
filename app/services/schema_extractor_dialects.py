"""PostgreSQL and MySQL BIRD-style schema extraction."""

from __future__ import annotations

import re
from typing import Any


def _normalize_pg_type(data_type: str) -> str:
    normalized = (data_type or "text").lower()
    if "int" in normalized or normalized == "serial":
        return "integer"
    if any(token in normalized for token in ("double", "float", "numeric", "decimal", "real")):
        return "real"
    if "date" in normalized or "time" in normalized:
        return "date"
    if "bool" in normalized:
        return "integer"
    return "text"


def _normalize_mysql_type(column_type: str) -> str:
    normalized = (column_type or "text").upper()
    if re.search(r"\b(INT|TINYINT|SMALLINT|MEDIUMINT|BIGINT)\b", normalized):
        return "integer"
    if re.search(r"\b(DECIMAL|NUMERIC|FLOAT|DOUBLE|REAL)\b", normalized):
        return "real"
    if re.search(r"\b(DATE|TIME|YEAR|DATETIME|TIMESTAMP)\b", normalized):
        return "date"
    if re.search(r"\bBOOL\b", normalized):
        return "integer"
    return "text"


def extract_postgres_schema(connection: Any, db_id: str) -> dict[str, Any]:
    cursor = connection.cursor()
    cursor.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
        ORDER BY table_name
        """
    )
    table_names_original = [row[0] for row in cursor.fetchall()]

    column_names_original: list[list[Any]] = [[-1, "*"]]
    column_names: list[list[Any]] = [[-1, "*"]]
    column_types: list[str] = []
    primary_keys: list[Any] = []
    foreign_keys: list[list[int]] = []
    column_index: dict[tuple[str, str], int] = {}

    for table_idx, table_name in enumerate(table_names_original):
        cursor.execute(
            """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = %s
            ORDER BY ordinal_position
            """,
            (table_name,),
        )
        columns = cursor.fetchall()
        pk_cols: list[int] = []

        for col_name, data_type in columns:
            global_idx = len(column_names_original)
            column_index[(table_name, col_name)] = global_idx
            column_names_original.append([table_idx, col_name])
            column_names.append([table_idx, col_name])
            column_types.append(_normalize_pg_type(data_type))

        cursor.execute(
            """
            SELECT kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name
             AND tc.table_schema = kcu.table_schema
            WHERE tc.constraint_type = 'PRIMARY KEY'
              AND tc.table_schema = 'public'
              AND tc.table_name = %s
            ORDER BY kcu.ordinal_position
            """,
            (table_name,),
        )
        for (pk_col,) in cursor.fetchall():
            pk_cols.append(column_index[(table_name, pk_col)])
        if pk_cols:
            primary_keys.append(pk_cols if len(pk_cols) > 1 else pk_cols[0])

    cursor.execute(
        """
        SELECT
            tc.table_name,
            kcu.column_name,
            ccu.table_name AS foreign_table_name,
            ccu.column_name AS foreign_column_name
        FROM information_schema.table_constraints AS tc
        JOIN information_schema.key_column_usage AS kcu
          ON tc.constraint_name = kcu.constraint_name
         AND tc.table_schema = kcu.table_schema
        JOIN information_schema.constraint_column_usage AS ccu
          ON ccu.constraint_name = tc.constraint_name
         AND ccu.table_schema = tc.table_schema
        WHERE tc.constraint_type = 'FOREIGN KEY'
          AND tc.table_schema = 'public'
        """
    )
    for table_name, col_name, ref_table, ref_col in cursor.fetchall():
        from_idx = column_index.get((table_name, col_name))
        to_idx = column_index.get((ref_table, ref_col))
        if from_idx is not None and to_idx is not None:
            foreign_keys.append([from_idx, to_idx])

    return {
        "db_id": db_id,
        "table_names_original": table_names_original,
        "table_names": table_names_original,
        "column_names_original": column_names_original,
        "column_names": column_names,
        "column_types": column_types,
        "primary_keys": primary_keys,
        "foreign_keys": foreign_keys,
    }


def extract_mysql_schema(connection: Any, db_id: str) -> dict[str, Any]:
    cursor = connection.cursor()
    cursor.execute("SHOW TABLES")
    table_names_original = [row[0] for row in cursor.fetchall()]

    column_names_original: list[list[Any]] = [[-1, "*"]]
    column_names: list[list[Any]] = [[-1, "*"]]
    column_types: list[str] = []
    primary_keys: list[Any] = []
    foreign_keys: list[list[int]] = []
    column_index: dict[tuple[str, str], int] = {}

    for table_idx, table_name in enumerate(table_names_original):
        cursor.execute(f"SHOW COLUMNS FROM `{table_name}`")
        columns = cursor.fetchall()
        pk_cols: list[int] = []

        for col_name, col_type, _null, key, _default, _extra in columns:
            global_idx = len(column_names_original)
            column_index[(table_name, col_name)] = global_idx
            column_names_original.append([table_idx, col_name])
            column_names.append([table_idx, col_name])
            column_types.append(_normalize_mysql_type(col_type))
            if key == "PRI":
                pk_cols.append(global_idx)

        if pk_cols:
            primary_keys.append(pk_cols if len(pk_cols) > 1 else pk_cols[0])

        cursor.execute(
            """
            SELECT COLUMN_NAME, REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
            FROM information_schema.KEY_COLUMN_USAGE
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = %s
              AND REFERENCED_TABLE_NAME IS NOT NULL
            """,
            (table_name,),
        )
        for col_name, ref_table, ref_col in cursor.fetchall():
            from_idx = column_index.get((table_name, col_name))
            to_idx = column_index.get((ref_table, ref_col))
            if from_idx is not None and to_idx is not None:
                foreign_keys.append([from_idx, to_idx])

    return {
        "db_id": db_id,
        "table_names_original": table_names_original,
        "table_names": table_names_original,
        "column_names_original": column_names_original,
        "column_names": column_names,
        "column_types": column_types,
        "primary_keys": primary_keys,
        "foreign_keys": foreign_keys,
    }
