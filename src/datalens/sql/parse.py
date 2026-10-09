"""Extracting SQL from model output and analysing it with sqlglot."""

from __future__ import annotations

import re
from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
# A fence may carry any language tag on its own line (sql, sqlite, postgresql, ...),
# or a sql/sqlite tag followed by the query on the same line.
_FENCE_RE = re.compile(r"```(?:[\w+-]*[ \t]*\r?\n|(?:sql|sqlite)[ \t]+)?(.*?)```", re.DOTALL | re.IGNORECASE)
_BARE_SQL_RE = re.compile(r"\b(WITH|SELECT)\b", re.IGNORECASE)

# Statement types that may modify a database or its connection state.
_WRITE_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.Command,
    exp.Merge,
)


def extract_sql(text: str) -> str | None:
    """Pull the final SQL query out of a model completion.

    Reasoning inside ``<think>...</think>`` is ignored, and output that stops
    inside an unclosed ``<think>`` has no answer. The last fenced code block
    wins, since models often sketch drafts before the answer. Without a fence,
    everything from the first ``SELECT``/``WITH`` that starts a line onwards is
    used. Returns ``None`` when nothing SQL-like is found.
    """
    if not text:
        return None
    body = _THINK_RE.sub("", text)
    # An unterminated <think> means generation stopped mid-reasoning (usually at
    # max_tokens): anything SQL-like in it is a draft, not an answer.
    if "<think>" in body.lower():
        return None

    blocks = [b.strip() for b in _FENCE_RE.findall(body) if b.strip()]
    if blocks:
        return _clean(blocks[-1])

    matches = list(_BARE_SQL_RE.finditer(body))
    for match in matches:
        prefix = body[: match.start()].rstrip(" \t")
        # Prefer a keyword that starts a line, not "select" mid-sentence. Taking the
        # first such keyword keeps line-leading subqueries inside the query.
        if not prefix.strip() or prefix.endswith(("\n", ":", ";")):
            return _clean(body[match.start() :])
    if matches:
        return _clean(body[matches[0].start() :])
    return None


def _clean(sql: str) -> str | None:
    sql = sql.strip().rstrip(";").strip()
    # Keep only the first statement if the model emitted several.
    if ";" in sql:
        sql = sql.split(";", 1)[0].strip()
    return sql or None


def is_read_only(sql: str) -> bool:
    """True if ``sql`` parses as exactly one query that cannot write.

    This is a defence-in-depth check for the API; the executor also opens every
    database read-only.
    """
    try:
        statements = [s for s in sqlglot.parse(sql, read="sqlite") if s is not None]
    except SqlglotError:  # ParseError, or TokenError for e.g. an unclosed quote
        return False
    if len(statements) != 1:
        return False
    statement = statements[0]
    if not isinstance(statement, exp.Query):
        return False
    return not any(isinstance(node, _WRITE_NODES) for node in statement.walk())


@dataclass(frozen=True)
class SchemaRefs:
    """Tables and columns a query mentions (lower-cased, unqualified)."""

    tables: frozenset[str]
    columns: frozenset[str]


def schema_refs(sql: str) -> SchemaRefs | None:
    """Tables and column names referenced by ``sql``; ``None`` if it won't parse.

    Columns are matched by name only (aliases like ``T1.name`` become ``name``),
    which is enough for schema-linking precision and recall.
    """
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except SqlglotError:
        return None
    if tree is None:
        return None
    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    tables = {t.name.lower() for t in tree.find_all(exp.Table) if t.name and t.name.lower() not in cte_names}
    columns = {c.name.lower() for c in tree.find_all(exp.Column) if c.name and c.name != "*"}
    # Output aliases (SELECT x AS total ... ORDER BY total) are not schema columns.
    aliases = {a.alias.lower() for a in tree.find_all(exp.Alias) if a.alias}
    return SchemaRefs(tables=frozenset(tables), columns=frozenset(columns - aliases))
