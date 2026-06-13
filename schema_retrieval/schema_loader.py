"""Parse the BIRD schema into structured table/column records.

Sources merged per database:
  * ``dev_tables.json`` - table names, column names (original + friendly),
    column types, primary keys, foreign keys.
  * ``database_description/<table>.csv`` - per-column natural-language
    descriptions, value descriptions and data formats.
  * the ``<db>.sqlite`` file - optional sampling of distinct values for text
    columns (helps the model match literals such as ``'EUR'``).
"""
from __future__ import annotations

import csv
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from config import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass
class ColumnRecord:
    db_id: str
    table: str
    name: str  # original column name (used in SQL)
    friendly_name: str
    col_type: str
    is_primary_key: bool = False
    is_foreign_key: bool = False
    description: str = ""
    value_description: str = ""
    data_format: str = ""
    sample_values: List[str] = field(default_factory=list)
    ordinal: int = 0  # position within its table (for stable ordering)
    global_index: int = -1  # index within dev_tables.json column arrays

    def embedding_text(self) -> str:
        """Rich text used to embed this column into the vector store."""
        parts = [f"{self.db_id}.{self.table}.{self.name}"]
        if self.friendly_name and self.friendly_name.lower() != self.name.lower():
            parts.append(f"({self.friendly_name})")
        parts.append(f"type={self.col_type}")
        if self.is_primary_key:
            parts.append("primary key")
        if self.description:
            parts.append(f"description: {self.description}")
        if self.value_description:
            parts.append(f"values: {self.value_description}")
        elif self.sample_values:
            parts.append("examples: " + ", ".join(self.sample_values))
        return " | ".join(parts)


@dataclass
class ForeignKey:
    db_id: str
    from_table: str
    from_column: str
    to_table: str
    to_column: str


@dataclass
class TableRecord:
    db_id: str
    name: str
    friendly_name: str
    table_index: int
    columns: List[ColumnRecord] = field(default_factory=list)

    def embedding_text(self) -> str:
        col_names = ", ".join(c.name for c in self.columns)
        label = self.name
        if self.friendly_name and self.friendly_name.lower() != self.name.lower():
            label = f"{self.name} ({self.friendly_name})"
        return f"table {label} | columns: {col_names}"


@dataclass
class DatabaseSchema:
    db_id: str
    tables: List[TableRecord] = field(default_factory=list)
    foreign_keys: List[ForeignKey] = field(default_factory=list)

    @property
    def all_columns(self) -> List[ColumnRecord]:
        return [c for t in self.tables for c in t.columns]


# ----------------------------------------------------------------------------
# Description CSV parsing
# ----------------------------------------------------------------------------
# BIRD description CSVs are a mix of UTF-8 (with/without BOM) and cp1252.
# utf-8-sig strips any BOM; cp1252 then latin-1 cover the legacy files.
_CSV_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")


def _read_csv_text(desc_path: Path) -> Optional[str]:
    for encoding in _CSV_ENCODINGS:
        try:
            return desc_path.read_text(encoding=encoding)
        except (UnicodeDecodeError, OSError):
            continue
    logger.warning("Could not decode description file %s", desc_path)
    return None


def _load_table_descriptions(desc_path: Path) -> Dict[str, Dict[str, str]]:
    """Map lower-cased original column name -> description fields."""
    if not desc_path.exists():
        return {}
    text = _read_csv_text(desc_path)
    if text is None:
        return {}

    out: Dict[str, Dict[str, str]] = {}
    try:
        reader = csv.DictReader(text.splitlines())
        for row in reader:
            norm = {
                (k or "").strip().lower(): (v or "").strip()
                for k, v in row.items()
                if k is not None
            }
            key = norm.get("original_column_name", "")
            if not key:
                continue
            out[key.lower()] = {
                "description": norm.get("column_description", ""),
                "value_description": norm.get("value_description", ""),
                "data_format": norm.get("data_format", ""),
            }
    except csv.Error as exc:
        logger.warning("Failed to parse description file %s: %s", desc_path, exc)
    return out


def _find_description_file(desc_dir: Path, table_name: str) -> Optional[Path]:
    """Locate a table's description CSV, tolerating case differences."""
    direct = desc_dir / f"{table_name}.csv"
    if direct.exists():
        return direct
    if not desc_dir.exists():
        return None
    target = f"{table_name.lower()}.csv"
    for candidate in desc_dir.glob("*.csv"):
        if candidate.name.lower() == target:
            return candidate
    return None


# ----------------------------------------------------------------------------
# Value sampling
# ----------------------------------------------------------------------------
def _sample_values_for_db(
    sqlite_path: Path,
    tables: List[TableRecord],
    settings: Settings,
) -> None:
    """Populate ``sample_values`` on text columns, mutating records in place."""
    if not settings.sample_values or not sqlite_path.exists():
        if settings.sample_values and not sqlite_path.exists():
            logger.warning("SQLite file not found, skipping sampling: %s", sqlite_path)
        return

    try:
        conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        logger.warning("Could not open %s for sampling: %s", sqlite_path, exc)
        return

    try:
        cursor = conn.cursor()
        for table in tables:
            for column in table.columns:
                if settings.sample_only_text and column.col_type.lower() not in {
                    "text",
                    "",
                }:
                    continue
                column.sample_values = _sample_column(
                    cursor, table.name, column.name, settings
                )
    finally:
        conn.close()


def _sample_column(
    cursor: sqlite3.Cursor,
    table: str,
    column: str,
    settings: Settings,
) -> List[str]:
    # Cap the scan first, then take distinct values to bound cost on large DBs.
    query = (
        f'SELECT DISTINCT "{column}" FROM '
        f'(SELECT "{column}" FROM "{table}" '
        f"LIMIT {int(settings.max_distinct_scan)}) "
        f'WHERE "{column}" IS NOT NULL '
        f"LIMIT {int(settings.max_sample_values)}"
    )
    try:
        cursor.execute(query)
        rows = cursor.fetchall()
    except sqlite3.Error:
        return []

    values: List[str] = []
    for (raw,) in rows:
        text = str(raw).strip()
        if not text:
            continue
        if len(text) > settings.max_value_char_len:
            text = text[: settings.max_value_char_len] + "..."
        values.append(text)
    return values


# ----------------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------------
def _flatten_primary_keys(primary_keys: List) -> set:
    flat: set = set()
    for entry in primary_keys:
        if isinstance(entry, list):
            flat.update(entry)
        else:
            flat.add(entry)
    return flat


def _build_database_schema(
    raw: dict,
    settings: Settings,
    *,
    sqlite_path: Optional[Path] = None,
    description_dir: Optional[Path] = None,
) -> DatabaseSchema:
    db_id = raw["db_id"]
    table_names = raw["table_names_original"]
    table_friendly = raw.get("table_names", table_names)
    columns_original = raw["column_names_original"]
    columns_friendly = raw.get("column_names", columns_original)
    column_types = raw.get("column_types", [])
    pk_indices = _flatten_primary_keys(raw.get("primary_keys", []))

    # Build table shells.
    tables: List[TableRecord] = [
        TableRecord(
            db_id=db_id,
            name=table_names[i],
            friendly_name=table_friendly[i] if i < len(table_friendly) else table_names[i],
            table_index=i,
        )
        for i in range(len(table_names))
    ]

    # Pre-load descriptions per table (overridable for arbitrary user DBs).
    desc_dir = description_dir if description_dir is not None else settings.description_dir(db_id)
    desc_cache: Dict[int, Dict[str, Dict[str, str]]] = {}
    for table in tables:
        path = _find_description_file(desc_dir, table.name)
        desc_cache[table.table_index] = (
            _load_table_descriptions(path) if path else {}
        )

    index_to_column: Dict[int, ColumnRecord] = {}
    for global_index, (table_idx, col_name) in enumerate(columns_original):
        if table_idx < 0:  # the synthetic "*" column
            continue
        friendly = (
            columns_friendly[global_index][1]
            if global_index < len(columns_friendly)
            else col_name
        )
        col_type = (
            column_types[global_index] if global_index < len(column_types) else ""
        )
        desc = desc_cache.get(table_idx, {}).get(col_name.lower(), {})
        record = ColumnRecord(
            db_id=db_id,
            table=table_names[table_idx],
            name=col_name,
            friendly_name=friendly,
            col_type=col_type or "",
            is_primary_key=global_index in pk_indices,
            description=desc.get("description", ""),
            value_description=desc.get("value_description", ""),
            data_format=desc.get("data_format", ""),
            ordinal=len(tables[table_idx].columns),
            global_index=global_index,
        )
        tables[table_idx].columns.append(record)
        index_to_column[global_index] = record

    # Foreign keys (indices reference the global column arrays).
    foreign_keys: List[ForeignKey] = []
    for from_idx, to_idx in raw.get("foreign_keys", []):
        from_col = index_to_column.get(from_idx)
        to_col = index_to_column.get(to_idx)
        if not from_col or not to_col:
            continue
        from_col.is_foreign_key = True
        foreign_keys.append(
            ForeignKey(
                db_id=db_id,
                from_table=from_col.table,
                from_column=from_col.name,
                to_table=to_col.table,
                to_column=to_col.name,
            )
        )

    effective_sqlite = sqlite_path if sqlite_path is not None else settings.sqlite_path(db_id)
    _sample_values_for_db(effective_sqlite, tables, settings)
    return DatabaseSchema(db_id=db_id, tables=tables, foreign_keys=foreign_keys)


def load_schemas(settings: Settings | None = None) -> List[DatabaseSchema]:
    """Load and assemble all database schemas from the BIRD dataset."""
    settings = settings or get_settings()
    tables_json = settings.tables_json_path
    if not tables_json.exists():
        raise FileNotFoundError(f"dev_tables.json not found at {tables_json}")

    logger.info("Loading schema definitions from %s", tables_json)
    with tables_json.open("r", encoding="utf-8") as fh:
        raw_schemas = json.load(fh)

    schemas = [_build_database_schema(raw, settings) for raw in raw_schemas]
    logger.info(
        "Loaded %d databases, %d tables, %d columns",
        len(schemas),
        sum(len(s.tables) for s in schemas),
        sum(len(s.all_columns) for s in schemas),
    )
    return schemas


def load_schema_from_files(
    schema_json: Path | str,
    sqlite_path: Optional[Path | str] = None,
    db_id: Optional[str] = None,
    description_dir: Optional[Path | str] = None,
    settings: Settings | None = None,
) -> List[DatabaseSchema]:
    """Load a schema from a user-supplied schema JSON + (optional) SQLite file.

    The JSON may follow the BIRD ``dev_tables.json`` format as either a single
    database object or a list of objects. When the JSON contains multiple
    databases, ``db_id`` must be provided to select which one to build (since a
    single ``sqlite_path`` is supplied for value sampling).
    """
    settings = settings or get_settings()
    schema_json = Path(schema_json)
    if not schema_json.exists():
        raise FileNotFoundError(f"Schema JSON not found at {schema_json}")

    sqlite_path = Path(sqlite_path) if sqlite_path is not None else None
    description_dir = Path(description_dir) if description_dir is not None else None

    logger.info("Loading schema definition from %s", schema_json)
    with schema_json.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    raw_list = raw if isinstance(raw, list) else [raw]
    if not raw_list:
        raise ValueError(f"Schema JSON {schema_json} contains no databases")

    available = [entry.get("db_id") for entry in raw_list]
    if db_id is not None:
        selected = [entry for entry in raw_list if entry.get("db_id") == db_id]
        if not selected:
            raise ValueError(
                f"db_id '{db_id}' not found in {schema_json}; available: {available}"
            )
    elif len(raw_list) == 1:
        selected = raw_list
    else:
        raise ValueError(
            f"Schema JSON {schema_json} contains multiple databases {available}; "
            "specify db_id to choose one."
        )

    schemas = [
        _build_database_schema(
            entry,
            settings,
            sqlite_path=sqlite_path,
            description_dir=description_dir,
        )
        for entry in selected
    ]
    logger.info(
        "Loaded db '%s': %d tables, %d columns",
        schemas[0].db_id,
        len(schemas[0].tables),
        len(schemas[0].all_columns),
    )
    return schemas
