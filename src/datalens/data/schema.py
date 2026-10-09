"""Turning a SQLite database into the schema text a model sees.

The rendering is DDL with inline comments, close to the OmniSQL / Arctic-R1 prompt
format: each column lists a few real example values (which helps the model match
literals such as ``'F'`` vs ``'female'``), and optionally BIRD's column
descriptions from ``database_description/*.csv``.
"""

from __future__ import annotations

import csv
import json
import re
import sqlite3
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

from datalens.sql.executor import connect_readonly, execute

_PLAIN_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def quote_ident(name: str) -> str:
    """Quote an identifier with backticks when it isn't a plain SQL name."""
    if _PLAIN_IDENT.match(name):
        return name
    return "`" + name.replace("`", "``") + "`"


@dataclass
class Column:
    name: str
    type: str
    primary_key: bool = False
    examples: list[str | int | float] = field(default_factory=list)
    description: str | None = None
    value_description: str | None = None


@dataclass
class ForeignKey:
    column: str
    ref_table: str
    ref_column: str


@dataclass
class Table:
    name: str
    columns: list[Column]
    foreign_keys: list[ForeignKey] = field(default_factory=list)


@dataclass
class DatabaseSchema:
    db_id: str
    tables: list[Table]

    def table_names(self) -> list[str]:
        return [t.name for t in self.tables]

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, text: str) -> DatabaseSchema:
        raw = json.loads(text)
        tables = [
            Table(
                name=t["name"],
                columns=[Column(**c) for c in t["columns"]],
                foreign_keys=[ForeignKey(**fk) for fk in t["foreign_keys"]],
            )
            for t in raw["tables"]
        ]
        return cls(db_id=raw["db_id"], tables=tables)

    def render(
        self,
        *,
        examples: bool = True,
        descriptions: bool = False,
        max_example_chars: int = 40,
        max_description_chars: int = 120,
    ) -> str:
        """Render ``CREATE TABLE`` statements with per-column comments."""
        blocks = []
        for table in self.tables:
            lines = []
            for col in table.columns:
                line = f"    {quote_ident(col.name)} {col.type or 'TEXT'},"
                notes = []
                if descriptions and col.description:
                    notes.append(_truncate(col.description, max_description_chars))
                if descriptions and col.value_description:
                    notes.append("values: " + _truncate(col.value_description, max_description_chars))
                if examples and col.examples:
                    shown = [
                        _truncate(v, max_example_chars) if isinstance(v, str) else v for v in col.examples
                    ]
                    notes.append("example: " + json.dumps(shown, ensure_ascii=False))
                if notes:
                    line += " -- " + "; ".join(notes)
                lines.append(line)
            pks = [quote_ident(c.name) for c in table.columns if c.primary_key]
            if pks:
                lines.append(f"    PRIMARY KEY ({', '.join(pks)}),")
            for fk in table.foreign_keys:
                lines.append(
                    f"    FOREIGN KEY ({quote_ident(fk.column)}) REFERENCES "
                    f"{quote_ident(fk.ref_table)} ({quote_ident(fk.ref_column)}),"
                )
            if lines:
                # Drop the trailing comma before any comment on the last line.
                last = lines[-1]
                head, sep, comment = last.partition(" -- ")
                lines[-1] = head.rstrip(",") + (sep + comment if sep else "")
            blocks.append(f"CREATE TABLE {quote_ident(table.name)} (\n" + "\n".join(lines) + "\n);")
        return "\n\n".join(blocks)


def _truncate(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _read_descriptions(desc_dir: Path, table: str) -> dict[str, tuple[str | None, str | None]]:
    """Column descriptions from BIRD's ``database_description/<table>.csv``."""
    candidates = [p for p in desc_dir.glob("*.csv") if p.stem.lower() == table.lower()]
    if not candidates:
        return {}
    out: dict[str, tuple[str | None, str | None]] = {}
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            with candidates[0].open(newline="", encoding=encoding) as fh:
                for row in csv.DictReader(fh):
                    name = (row.get("original_column_name") or "").strip()
                    if not name:
                        continue
                    desc = (row.get("column_description") or "").strip() or None
                    value_desc = (row.get("value_description") or "").strip() or None
                    # Skip descriptions that just restate the column name.
                    if desc and _normalise(desc) == _normalise(name):
                        desc = None
                    out[name.lower()] = (desc, value_desc)
            return out
        except UnicodeDecodeError:
            out.clear()
    return out


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def introspect_sqlite(
    db_path: str | Path,
    db_id: str | None = None,
    *,
    num_examples: int = 3,
    example_timeout_s: float = 2.0,
) -> DatabaseSchema:
    """Read tables, columns, keys and example values from a SQLite file.

    Example values come from ``SELECT DISTINCT ... LIMIT n`` with a short timeout,
    so very large tables cost at most ``example_timeout_s`` per column.
    """
    db_path = Path(db_path)
    db_id = db_id or db_path.stem
    desc_dir = db_path.parent / "database_description"

    conn = connect_readonly(db_path)
    try:
        table_names = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY rowid"
            )
        ]
        tables = []
        for name in table_names:
            info = conn.execute(f"PRAGMA table_info({quote_ident(name)})").fetchall()
            fk_rows = conn.execute(f"PRAGMA foreign_key_list({quote_ident(name)})").fetchall()
            descriptions = _read_descriptions(desc_dir, name) if desc_dir.is_dir() else {}
            columns = []
            for _cid, col_name, col_type, _notnull, _default, pk in info:
                desc, value_desc = descriptions.get(col_name.lower(), (None, None))
                columns.append(
                    Column(
                        name=col_name,
                        type=(col_type or "").upper(),
                        primary_key=bool(pk),
                        description=desc,
                        value_description=value_desc,
                    )
                )
            foreign_keys = [
                ForeignKey(column=row[3], ref_table=row[2], ref_column=row[4] or row[3]) for row in fk_rows
            ]
            tables.append(Table(name=name, columns=columns, foreign_keys=foreign_keys))
    except sqlite3.Error as exc:
        raise ValueError(f"could not read schema of {db_path}: {exc}") from exc
    finally:
        conn.close()

    if num_examples > 0:
        for table in tables:
            for col in table.columns:
                col.examples = _example_values(db_path, table.name, col.name, num_examples, example_timeout_s)
    return DatabaseSchema(db_id=db_id, tables=tables)


def _example_values(
    db_path: Path, table: str, column: str, limit: int, timeout_s: float
) -> list[str | int | float]:
    col = quote_ident(column)
    sql = (
        f"SELECT DISTINCT {col} FROM {quote_ident(table)} "
        f"WHERE {col} IS NOT NULL AND CAST({col} AS TEXT) != '' LIMIT {int(limit)}"
    )
    result = execute(db_path, sql, timeout_s=timeout_s)
    if not result.ok or not result.rows:
        return []
    values: list[str | int | float] = []
    for (value,) in result.rows:
        if isinstance(value, bytes):
            continue  # BLOBs carry no useful signal for the model
        # Keep numbers as numbers so the model doesn't quote numeric literals.
        values.append(value if isinstance(value, (int, float)) else str(value))
    return values


@lru_cache(maxsize=256)
def _cached_schema(db_path: str, mtime: float, num_examples: int) -> DatabaseSchema:
    del mtime  # part of the cache key only
    return introspect_sqlite(db_path, num_examples=num_examples)


def load_schema(
    db_path: str | Path,
    *,
    num_examples: int = 3,
    cache_dir: str | Path | None = None,
) -> DatabaseSchema:
    """Introspect ``db_path`` with in-memory and optional on-disk caching.

    Introspection of a large BIRD database takes seconds, and every example in a
    training or evaluation run needs one, so results are cached per file.
    """
    path = Path(db_path).resolve()
    mtime = path.stat().st_mtime
    if cache_dir is not None:
        cache_file = Path(cache_dir) / f"{path.stem}-{int(mtime)}-ex{num_examples}.json"
        if cache_file.is_file():
            return DatabaseSchema.from_json(cache_file.read_text(encoding="utf-8"))
        schema = _cached_schema(str(path), mtime, num_examples)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(schema.to_json(), encoding="utf-8")
        return schema
    return _cached_schema(str(path), mtime, num_examples)
