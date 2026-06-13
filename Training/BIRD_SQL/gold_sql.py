"""Extract the tables referenced by a gold SQL query.

Uses ``sqlglot`` to parse the (SQLite-dialect) gold SQL and collect every
referenced base table, resolving them case-insensitively to the real schema
table names. Aliases are handled natively: ``sqlglot`` exposes the underlying
table name on :class:`sqlglot.exp.Table`, not the alias.

This powers two things:
  * ``gold_union`` context construction (guarantee the gold tables are present),
  * the ``--report-coverage`` self-check (did the rendered context include every
    table the gold SQL needs?).
"""
from __future__ import annotations

import logging
from typing import Iterable, Set

logger = logging.getLogger(__name__)


def _require_sqlglot():
    try:
        import sqlglot
        from sqlglot import exp
    except ImportError as exc:  # pragma: no cover - exercised only when missing
        raise ImportError(
            "sqlglot is required for gold-SQL parsing (gold_union strategy or "
            "--report-coverage). Install it with: pip install sqlglot"
        ) from exc
    return sqlglot, exp


def extract_referenced_tables(sql: str, known_tables: Iterable[str]) -> Set[str]:
    """Return the subset of ``known_tables`` referenced by ``sql``.

    Args:
        sql: the gold SQL query (SQLite dialect for BIRD).
        known_tables: the real table names of the target database.

    Returns:
        A set of real table names (as they appear in ``known_tables``) that the
        query references. Unparseable queries yield an empty set (logged).
    """
    if not sql or not sql.strip():
        return set()

    sqlglot, exp = _require_sqlglot()
    name_map = {t.lower(): t for t in known_tables}

    try:
        expressions = sqlglot.parse(sql, read="sqlite")
    except Exception:  # noqa: BLE001 - sqlglot raises a variety of errors
        try:
            expressions = [sqlglot.parse_one(sql, read="sqlite")]
        except Exception:  # noqa: BLE001
            logger.warning("Could not parse gold SQL for table extraction: %s", sql)
            return set()

    found: Set[str] = set()
    for expression in expressions:
        if expression is None:
            continue
        for table in expression.find_all(exp.Table):
            real = name_map.get((table.name or "").lower())
            if real is not None:
                found.add(real)
    return found
