"""Running candidate queries and describing their results to the model.

Every candidate is executed read-only (through :func:`datalens.sql.executor.execute`)
and summarised as an :class:`Outcome`: whether it ran, the result fingerprint
used to group candidates, and a short text preview of the rows, which is what
the fixer, reviser and selector see as "execution result".
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from datalens.sql.compare import result_key
from datalens.sql.executor import execute

EMPTY_RESULT = "Empty result: the query returned no rows."


@dataclass(frozen=True)
class Outcome:
    """What running one query produced."""

    ok: bool
    key: str | None = None  # result fingerprint; equal keys mean equal result sets
    n_rows: int = 0
    empty: bool = False
    error: str | None = None
    error_kind: str | None = None
    preview: str = ""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Outcome:
        return cls(**data)

    def describe(self) -> str:
        """The "Execution Result" text shown in prompts."""
        if not self.ok:
            return f"Error: {self.error}"
        return EMPTY_RESULT if self.empty else self.preview


def _cell(value: Any, limit: int) -> str:
    text = "NULL" if value is None else " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def render_rows(
    columns: Sequence[str], rows: Sequence[tuple[Any, ...]], *, max_rows: int = 10, max_cell: int = 60
) -> str:
    """A compact, pipe-separated preview of a result set."""
    header = " | ".join(_cell(c, max_cell) for c in columns) if columns else ""
    lines = [f"Columns: {header}", f"Rows: {len(rows)}"]
    lines += [" | ".join(_cell(v, max_cell) for v in row) for row in rows[:max_rows]]
    if len(rows) > max_rows:
        lines.append(f"... ({len(rows) - max_rows} more rows)")
    return "\n".join(lines)


def run_query(
    db_path: str | Path,
    sql: str | None,
    *,
    timeout_s: float = 30.0,
    preview_rows: int = 10,
    max_rows: int = 100_000,
) -> Outcome:
    """Execute ``sql`` and summarise the result."""
    if not sql:
        return Outcome(ok=False, error="no SQL query found in the model output", error_kind="no_sql")
    result = execute(db_path, sql, timeout_s=timeout_s, max_rows=max_rows)
    if not result.ok or result.rows is None:
        kind = result.error_kind.value if result.error_kind is not None else "runtime_error"
        return Outcome(ok=False, error=result.error, error_kind=kind)
    key = result_key(result.rows)
    if result.truncated:
        # Only a prefix was fetched, so this fingerprint must not merge with full results.
        key = "truncated:" + key
    return Outcome(
        ok=True,
        key=key,
        n_rows=len(result.rows),
        empty=not result.rows,
        preview=render_rows(result.columns or [], result.rows, max_rows=preview_rows),
    )


class QueryRunner:
    """Runs many queries in a thread pool, executing each distinct (db, sql) once."""

    def __init__(self, *, timeout_s: float = 30.0, preview_rows: int = 10, workers: int = 8) -> None:
        self.timeout_s = timeout_s
        self.preview_rows = preview_rows
        self.workers = workers

    def run_many(self, jobs: Sequence[tuple[str | Path, str | None]]) -> list[Outcome]:
        unique = list(dict.fromkeys((str(db), sql) for db, sql in jobs))

        def one(job: tuple[str, str | None]) -> Outcome:
            return run_query(job[0], job[1], timeout_s=self.timeout_s, preview_rows=self.preview_rows)

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            outcomes = dict(zip(unique, pool.map(one, unique), strict=True))
        return [outcomes[(str(db), sql)] for db, sql in jobs]
