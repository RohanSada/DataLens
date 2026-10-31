"""Finding literal values in a database: the agent's ``<search>`` tool and value hints.

Most wrong BIRD answers that *run* fail on a literal: the question says
"Fremont" and the column holds "Fremont Unified", or "female" where the data says
"F". The schema shows only three example values per column, so the agent needs a
way to look values up.

The index holds the distinct short text values of every text column, normalised
to lower case. Lookups are dictionary and token-index hits, so they are cheap
enough to run for every n-gram of every question (value hints) and for every
``<search>`` the agent issues. Indexes are built once per database file and
cached on disk.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from datalens.data.schema import quote_ident
from datalens.sql.executor import connect_readonly, execute

log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_TEXT_AFFINITY = ("CHAR", "CLOB", "TEXT", "")  # SQLite rules: '' (no type) also stores text

# Question words that are never worth looking up as values.
_STOPWORD_TEXT = """
    a about above after all also among an and any are as at be been being between both but by can
    could did do does each either for from had has have having he her hers him his how i if in into
    is it its least list many more most much name names number of on one only or other our out over
    per please provide same she should show so some state such than that the their them then there
    these they this those through to total under value values was we were what when where which
    while who whom whose why will with within would year years you your average count percentage
    ratio rate amount highest lowest top give tell find calculate
"""
_STOPWORDS = frozenset(_STOPWORD_TEXT.split())


def normalise(text: str) -> str:
    return " ".join(str(text).lower().split())


def tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(str(text).lower())


@dataclass(frozen=True)
class ValueMatch:
    table: str
    column: str
    value: str
    score: float  # 1.0 exact (case-insensitive), lower for partial/fuzzy matches

    @property
    def ref(self) -> str:
        return f"{quote_ident(self.table)}.{quote_ident(self.column)}"


class ValueIndex:
    """Distinct text values of one database, searchable by exact, token and fuzzy match."""

    def __init__(self, columns: list[tuple[str, str, list[str]]]) -> None:
        self.columns = [(t, c) for t, c, _ in columns]
        self.entries: list[tuple[int, str]] = [
            (ci, value) for ci, (_t, _c, values) in enumerate(columns) for value in values
        ]
        self.by_norm: dict[str, list[int]] = {}
        self.by_token: dict[str, list[int]] = {}
        for i, (_ci, value) in enumerate(self.entries):
            self.by_norm.setdefault(normalise(value), []).append(i)
            for tok in set(tokens(value)):
                self.by_token.setdefault(tok, []).append(i)
        self._vocab: dict[str, list[str]] | None = None

    def __len__(self) -> int:
        return len(self.entries)

    # ------------------------------------------------------------ building

    @classmethod
    def build(
        cls,
        db_path: str | Path,
        *,
        max_values_per_column: int = 50_000,
        max_value_chars: int = 80,
        timeout_s: float = 20.0,
    ) -> ValueIndex:
        db_path = Path(db_path)
        conn = connect_readonly(db_path)
        try:
            tables = [
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
                )
            ]
            text_columns = []
            for table in tables:
                for _cid, name, col_type, *_rest in conn.execute(f"PRAGMA table_info({quote_ident(table)})"):
                    upper = (col_type or "").upper()
                    if upper == "" or any(a and a in upper for a in _TEXT_AFFINITY):
                        text_columns.append((table, name))
        except sqlite3.Error as exc:
            raise ValueError(f"could not read {db_path}: {exc}") from exc
        finally:
            conn.close()

        columns: list[tuple[str, str, list[str]]] = []
        for table, column in text_columns:
            col = quote_ident(column)
            sql = (
                f"SELECT DISTINCT {col} FROM {quote_ident(table)} WHERE {col} IS NOT NULL "
                f"AND typeof({col}) = 'text' AND length({col}) BETWEEN 1 AND {int(max_value_chars)} "
                f"LIMIT {int(max_values_per_column)}"
            )
            result = execute(db_path, sql, timeout_s=timeout_s)
            if result.ok and result.rows:
                values = [str(v) for (v,) in result.rows if str(v).strip()]
                if values:
                    columns.append((table, column, values))
        return cls(columns)

    @classmethod
    def load(cls, db_path: str | Path, cache_dir: str | Path | None = None) -> ValueIndex:
        """Build, or load from ``cache_dir`` if this database file was indexed before."""
        path = Path(db_path).resolve()
        if cache_dir is None:
            return cls.build(path)
        cache_file = Path(cache_dir) / f"{path.stem}-{int(path.stat().st_mtime)}.json"
        if cache_file.is_file():
            raw = json.loads(cache_file.read_text(encoding="utf-8"))
            return cls([(t, c, values) for t, c, values in raw["columns"]])
        log.info("indexing values of %s", path.name)
        index = cls.build(path)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {"columns": index.column_values()}
        cache_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return index

    def column_values(self) -> list[tuple[str, str, list[str]]]:
        grouped: dict[int, list[str]] = {}
        for ci, value in self.entries:
            grouped.setdefault(ci, []).append(value)
        return [(*self.columns[ci], values) for ci, values in grouped.items()]

    # ------------------------------------------------------------ lookups

    def _match(self, i: int, score: float) -> ValueMatch:
        ci, value = self.entries[i]
        table, column = self.columns[ci]
        return ValueMatch(table, column, value, score)

    def exact(self, text: str) -> list[ValueMatch]:
        return [self._match(i, 1.0) for i in self.by_norm.get(normalise(text), [])]

    def _close_tokens(self, token: str, n: int = 3) -> list[str]:
        """Vocabulary tokens spelled like ``token`` (same first character, for speed)."""
        if self._vocab is None:
            self._vocab = {}
            for tok in self.by_token:
                self._vocab.setdefault(tok[0], []).append(tok)
        return difflib.get_close_matches(token, self._vocab.get(token[0], []), n=n, cutoff=0.8)

    def search(
        self,
        text: str,
        *,
        limit: int = 12,
        table: str | None = None,
        column: str | None = None,
        max_candidates: int = 3_000,
    ) -> list[ValueMatch]:
        """Values that equal, contain, or closely resemble ``text``, best first.

        Exact (case-insensitive) matches score 1.0, values containing every query
        token 0.5-1.0, values containing some of them or a similarly spelled token
        below that. Every step is an index lookup, so a search costs milliseconds
        even on databases with millions of values.
        """
        query = normalise(text)
        if not query:
            return []

        def wanted(i: int) -> bool:
            t, c = self.columns[self.entries[i][0]]
            return (table is None or t.lower() == table.lower()) and (
                column is None or c.lower() == column.lower()
            )

        def similarity(i: int) -> float:
            return difflib.SequenceMatcher(None, query, normalise(self.entries[i][1])).ratio()

        scored: dict[int, float] = {i: 1.0 for i in self.by_norm.get(query, []) if wanted(i)}
        query_tokens = list(dict.fromkeys(tokens(query)))
        postings = {t: self.by_token.get(t, []) for t in query_tokens}

        # Values that contain every query token.
        if query_tokens and all(postings.values()):
            ordered = sorted(postings.values(), key=len)
            candidates = set(ordered[0])
            for plist in ordered[1:]:
                candidates &= set(plist)
                if not candidates:
                    break
            for i in sorted(candidates)[:max_candidates]:
                if i not in scored and wanted(i):
                    scored[i] = 0.5 + 0.49 * similarity(i)

        if len(scored) < limit:
            # Values that contain some query tokens, or tokens spelled like them.
            partial: set[int] = set()
            for tok in query_tokens:
                options = [tok] if postings[tok] else []
                if len(tok) >= 4:
                    options += [t for t in self._close_tokens(tok) if t != tok]
                for option in options:
                    plist = self.by_token.get(option, [])
                    if len(plist) <= max_candidates:
                        partial.update(plist)
            for i in sorted(partial)[:max_candidates]:
                if i not in scored and wanted(i):
                    ratio = similarity(i)
                    if ratio >= 0.5:
                        scored[i] = 0.1 + 0.39 * ratio

        best = sorted(scored.items(), key=lambda kv: (-kv[1], len(self.entries[kv[0]][1]), kv[0]))
        return [self._match(i, s) for i, s in best[:limit]]

    def hints(self, question: str, evidence: str = "", *, limit: int = 12, max_columns: int = 4) -> list[str]:
        """Phrases from the question that appear verbatim as values, with where they appear.

        Every 1-4 word n-gram (and every quoted string) is looked up exactly. Longer
        phrases win over the words inside them, and phrases found in too many
        columns are skipped as uninformative.
        """
        text = f"{question} {evidence}"
        words = re.findall(r"[\w'.&/-]+", text)
        phrases: list[str] = re.findall(r"['\"]([^'\"]{2,80})['\"]", text)
        for n in (4, 3, 2, 1):
            for i in range(len(words) - n + 1):
                gram = " ".join(words[i : i + n]).strip(".,;:!?'\"()")
                if not gram or (n == 1 and (gram.lower() in _STOPWORDS or len(gram) < 3)):
                    continue
                if n > 1 and all(w.lower() in _STOPWORDS for w in words[i : i + n]):
                    continue
                phrases.append(gram)

        lines: list[str] = []
        covered: list[str] = []
        seen: set[str] = set()
        for phrase in phrases:
            key = normalise(phrase)
            if key in seen or any(key in c for c in covered):
                continue
            seen.add(key)
            matches = self.exact(phrase)
            refs = sorted({m.ref for m in matches})
            if not refs or len(refs) > max_columns:
                continue
            covered.append(key)
            values = sorted({m.value for m in matches})
            shown = ", ".join(json.dumps(v, ensure_ascii=False) for v in values[:3])
            lines.append(f"{shown} appears in {', '.join(refs)}")
            if len(lines) >= limit:
                break
        return lines


def format_matches(matches: list[ValueMatch], query: str) -> str:
    if not matches:
        return f"No text values similar to {json.dumps(query, ensure_ascii=False)} were found."
    by_ref: dict[str, list[str]] = {}
    for m in matches:
        by_ref.setdefault(m.ref, []).append(json.dumps(m.value, ensure_ascii=False))
    lines = [f"Values matching {json.dumps(query, ensure_ascii=False)}:"]
    lines.extend(f"- {ref}: {', '.join(vals)}" for ref, vals in by_ref.items())
    return "\n".join(lines)
