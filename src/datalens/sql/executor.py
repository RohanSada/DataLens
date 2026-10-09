"""Read-only, time-bounded SQLite execution.

Every query the system runs (rewards during RL, evaluation, the API) goes through
:func:`execute`. Databases are opened with ``mode=ro`` so a generated ``DROP`` or
``UPDATE`` cannot change them, and a SQLite progress handler aborts queries that
run past their deadline without needing a subprocess per query.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

Row = tuple[Any, ...]


class ErrorKind(str, Enum):
    """Coarse failure classes, used for the error taxonomy and API responses."""

    SYNTAX = "syntax_error"
    SCHEMA = "schema_error"
    TIMEOUT = "timeout"
    RUNTIME = "runtime_error"


@dataclass(frozen=True)
class ExecResult:
    """Outcome of running one statement."""

    ok: bool
    rows: list[Row] | None = None
    columns: list[str] | None = None
    error: str | None = None
    error_kind: ErrorKind | None = None
    elapsed_s: float = 0.0
    truncated: bool = False


_SCHEMA_MARKERS = ("no such table", "no such column", "ambiguous column name")
_SYNTAX_MARKERS = (
    "syntax error",
    "incomplete input",
    "unrecognized token",
    'near "',
    "you can only execute one statement",
)


def classify_error(message: str) -> ErrorKind:
    """Map a SQLite error message onto an :class:`ErrorKind`."""
    lowered = message.lower()
    if "interrupted" in lowered:
        return ErrorKind.TIMEOUT
    if any(marker in lowered for marker in _SCHEMA_MARKERS):
        return ErrorKind.SCHEMA
    if any(marker in lowered for marker in _SYNTAX_MARKERS):
        return ErrorKind.SYNTAX
    return ErrorKind.RUNTIME


def _decode(value: bytes) -> str:
    return value.decode("utf-8", errors="replace")


def connect_readonly(db_path: str | Path) -> sqlite3.Connection:
    """Open a SQLite database read-only. Raises ``FileNotFoundError`` if missing."""
    path = Path(db_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"database not found: {path}")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
    # Some BIRD databases contain invalid UTF-8; decode leniently and consistently
    # so gold and predicted results compare on equal terms.
    conn.text_factory = _decode
    return conn


def execute(
    db_path: str | Path,
    sql: str,
    *,
    timeout_s: float = 30.0,
    max_rows: int | None = None,
) -> ExecResult:
    """Run ``sql`` against ``db_path`` read-only, aborting after ``timeout_s``.

    ``max_rows`` caps how many rows are fetched (``truncated`` is set when the cap
    is hit). Evaluation leaves it unset so results compare exactly; the API sets it
    to keep responses small.
    """
    start = time.perf_counter()
    try:
        conn = connect_readonly(db_path)
    except (FileNotFoundError, sqlite3.Error) as exc:
        return ExecResult(
            ok=False,
            error=str(exc),
            error_kind=ErrorKind.RUNTIME,
            elapsed_s=time.perf_counter() - start,
        )

    deadline = start + timeout_s
    # Called every N SQLite VM instructions; a non-zero return aborts the query.
    conn.set_progress_handler(lambda: int(time.perf_counter() > deadline), 10_000)
    try:
        cursor = conn.execute(sql)
        columns = [d[0] for d in cursor.description] if cursor.description else []
        if max_rows is None:
            rows = cursor.fetchall()
            truncated = False
        else:
            rows = cursor.fetchmany(max_rows + 1)
            truncated = len(rows) > max_rows
            rows = rows[:max_rows]
        return ExecResult(
            ok=True,
            rows=[tuple(r) for r in rows],
            columns=columns,
            elapsed_s=time.perf_counter() - start,
            truncated=truncated,
        )
    except sqlite3.Error as exc:
        message = str(exc)
        kind = classify_error(message)
        if kind is ErrorKind.TIMEOUT:
            message = f"query exceeded {timeout_s:g}s timeout"
        return ExecResult(
            ok=False,
            error=message,
            error_kind=kind,
            elapsed_s=time.perf_counter() - start,
        )
    except (ValueError, OverflowError, MemoryError) as exc:  # e.g. huge ints, bad params
        return ExecResult(
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
            error_kind=ErrorKind.RUNTIME,
            elapsed_s=time.perf_counter() - start,
        )
    finally:
        conn.close()
