"""Database values a question mentions ("matched contents").

Agentar-Scale-SQL, like OmniSQL before it, looks up the cell values a question
refers to and shows them to the model next to their column, so a query filters
on ``'Chinese Grand Prix'`` rather than a guess at the spelling. This follows
the released OmniSQL / Agentar recipe:

* index every distinct, non-numeric string value of at most 40 characters;
* a value matches when more than 85% of it appears verbatim (ignoring case) in
  the evidence plus the question; matches rank by that share, then by length,
  and the top 20 are kept.

The released code finds candidate values with a Lucene BM25 index over n-grams
of the question. Here an inverted index over word tokens does the same job
without a Java dependency. Literals that the task-understanding model extracts
are also matched more loosely (edit similarity), which catches case and spelling
variants that are not verbatim in the question.
"""

from __future__ import annotations

import logging
import math
import pickle
import re
from array import array
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

from datalens.data.schema import quote_ident
from datalens.sql.executor import connect_readonly, execute

log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[0-9a-z]+")
# Function words that would make almost every value a candidate.
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "i",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "them",
        "there",
        "these",
        "they",
        "this",
        "those",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        "why",
        "with",
    ]
)
INDEX_VERSION = 1


def tokens(text: str) -> list[str]:
    """Lower-case alphanumeric word tokens."""
    return _TOKEN_RE.findall(text.lower())


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def match_ratio(value: str, text: str, floor: float) -> float:
    """Share of ``value`` covered by its longest substring that occurs in ``text``.

    Both strings should be lower case. Only substring lengths down to
    ``floor * len(value)`` are tried, so the result is 0.0 whenever the share
    would be below ``floor``; this keeps the check to a handful of substring
    searches per value.
    """
    n = len(value)
    if n == 0:
        return 0.0
    if value in text:
        return 1.0
    shortest = max(1, math.ceil(floor * n))
    for length in range(n - 1, shortest - 1, -1):
        for start in range(n - length + 1):
            if value[start : start + length] in text:
                return length / n
    return 0.0


def _variants(word: str) -> list[str]:
    """``word`` and its stems without 1-3 trailing letters, so plurals and inflections still
    find a value (the BM25 index of the released code stems words for the same reason)."""
    out = [word]
    for cut in (1, 2, 3):
        if len(word) - cut >= 3 and not word[-cut:].isdigit():
            out.append(word[:-cut])
    return out


def _whole_word(value: str, text: str) -> bool:
    return re.search(rf"(?<![0-9a-z]){re.escape(value)}(?![0-9a-z])", text) is not None


@dataclass(frozen=True)
class ValueMatch:
    table: str
    column: str
    value: str
    score: float

    @property
    def key(self) -> str:
        """``table.column`` in lower case, the key schema rendering expects."""
        return f"{self.table}.{self.column}".lower()


class ValueIndex:
    """Distinct string values of one database, with a token index for lookup."""

    def __init__(
        self,
        db_id: str,
        columns: list[tuple[str, str]],
        values: list[str],
        column_of: Sequence[int],
        *,
        max_postings: int = 20_000,
    ) -> None:
        self.db_id = db_id
        self.columns = columns
        self.values = values
        self.column_of = array("I", column_of)
        self.max_postings = max_postings
        self._lower = [v.lower() for v in values]
        postings: dict[str, list[int]] = {}
        for vid, value in enumerate(self._lower):
            for tok in set(tokens(value)):
                if tok not in STOPWORDS:
                    postings.setdefault(tok, []).append(vid)
        self.postings = {tok: array("I", ids) for tok, ids in postings.items()}

    def __len__(self) -> int:
        return len(self.values)

    @classmethod
    def build(
        cls,
        db_path: str | Path,
        *,
        max_chars: int = 40,
        max_values_per_column: int = 200_000,
        timeout_s: float = 120.0,
    ) -> ValueIndex:
        """Read the distinct, non-numeric string values of every column."""
        db_path = Path(db_path)
        conn = connect_readonly(db_path)
        try:
            tables = [
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
                )
            ]
            layout = [
                (table, row[1])
                for table in tables
                for row in conn.execute(f"PRAGMA table_info({quote_ident(table)})").fetchall()
            ]
        finally:
            conn.close()

        columns: list[tuple[str, str]] = []
        values: list[str] = []
        column_of: list[int] = []
        for table, column in layout:
            col = quote_ident(column)
            sql = (
                f"SELECT DISTINCT {col} FROM {quote_ident(table)} WHERE {col} IS NOT NULL "
                f"LIMIT {int(max_values_per_column)}"
            )
            result = execute(db_path, sql, timeout_s=timeout_s)
            if not result.ok or not result.rows:
                continue
            kept = []
            for (raw,) in result.rows:
                if not isinstance(raw, str):
                    continue
                text = raw.strip()
                if 0 < len(text) <= max_chars and not _is_number(text):
                    kept.append(text)
            if not kept:
                continue
            col_id = len(columns)
            columns.append((table, column))
            for text in dict.fromkeys(kept):
                values.append(text)
                column_of.append(col_id)
        log.info("indexed %d values from %d columns of %s", len(values), len(columns), db_path.name)
        return cls(db_path.stem, columns, values, column_of)

    def _candidates(self, words: Iterable[str]) -> set[int]:
        """Values sharing a word with ``words``, allowing inflected forms ("lamps" finds "Lamp")."""
        found: set[int] = set()
        for word in words:
            for tok in _variants(word):
                ids = self.postings.get(tok)
                if ids is not None and len(ids) <= self.max_postings:
                    found.update(ids)
        return found

    def _match(self, vid: int, score: float) -> ValueMatch:
        table, column = self.columns[self.column_of[vid]]
        return ValueMatch(table=table, column=column, value=self.values[vid], score=score)

    def match_text(self, text: str, *, threshold: float = 0.85, limit: int = 20) -> list[ValueMatch]:
        """Values that (almost) appear verbatim in ``text`` (OmniSQL's rule)."""
        lowered = text.lower()
        words = set(tokens(lowered)) - STOPWORDS
        scored: list[tuple[float, int, int]] = []
        for vid in self._candidates(words):
            value = self._lower[vid]
            if len(value) <= 3:
                # A short value like 'CA' or 'F' occurs inside many words; require it as a word.
                score = 1.0 if _whole_word(value, lowered) else 0.0
            else:
                score = match_ratio(value, lowered, threshold)
            if score > threshold:
                scored.append((score, len(value), vid))
        scored.sort(key=lambda s: (-s[0], -s[1], s[2]))
        return [self._match(vid, score) for score, _, vid in scored[:limit]]

    def match_literals(
        self,
        literals: Sequence[str],
        *,
        threshold: float = 0.8,
        per_literal: int = 3,
        max_candidates: int = 400,
    ) -> list[ValueMatch]:
        """Closest values to each extracted literal by edit similarity."""
        out: list[ValueMatch] = []
        for literal in literals:
            target = " ".join(literal.lower().split())
            words = set(tokens(target)) - STOPWORDS
            if not target or not words:
                continue
            pool = self._candidates(words)
            if len(pool) > max_candidates:
                # Keep the values that share the most words with the literal.
                ranked = sorted(pool, key=lambda v: -len(words & set(tokens(self._lower[v]))))
                pool = set(ranked[:max_candidates])
            scored = []
            for vid in pool:
                matcher = SequenceMatcher(None, target, self._lower[vid], autojunk=False)
                if matcher.real_quick_ratio() < threshold or matcher.quick_ratio() < threshold:
                    continue
                ratio = matcher.ratio()
                if ratio >= threshold:
                    scored.append((ratio, vid))
            scored.sort(key=lambda s: (-s[0], s[1]))
            out.extend(self._match(vid, ratio) for ratio, vid in scored[:per_literal])
        return out

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = (INDEX_VERSION, self.db_id, self.columns, self.values, self.column_of)
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> ValueIndex:
        with Path(path).open("rb") as fh:
            version, db_id, columns, values, column_of = pickle.load(fh)
        if version != INDEX_VERSION:
            raise ValueError(f"{path}: value index version {version}, expected {INDEX_VERSION}")
        return cls(db_id, columns, values, column_of)


def find_values(
    index: ValueIndex,
    text: str,
    literals: Sequence[str] = (),
    *,
    threshold: float = 0.85,
    limit: int = 20,
    literal_threshold: float = 0.8,
    per_literal: int = 3,
) -> list[ValueMatch]:
    """Matched contents for one question: verbatim matches, then literal look-ups."""
    hits = index.match_text(text, threshold=threshold, limit=limit)
    seen = {(h.key, h.value) for h in hits}
    for hit in index.match_literals(literals, threshold=literal_threshold, per_literal=per_literal):
        if (hit.key, hit.value) not in seen:
            seen.add((hit.key, hit.value))
            hits.append(hit)
    return hits


def group_matches(matches: Iterable[ValueMatch]) -> dict[str, list[str]]:
    """``{"table.column": [values]}`` in match order, for schema rendering."""
    grouped: dict[str, list[str]] = {}
    for m in matches:
        values = grouped.setdefault(m.key, [])
        if m.value not in values:
            values.append(m.value)
    return grouped


def render_matches(matches: Iterable[ValueMatch]) -> str:
    """Matched contents as text, one ``table.column: values`` line per column."""
    grouped: dict[tuple[str, str], list[str]] = {}
    for m in matches:
        values = grouped.setdefault((m.table, m.column), [])
        if m.value not in values:
            values.append(m.value)
    if not grouped:
        return "None"
    return "\n".join(
        f"{quote_ident(table)}.{quote_ident(column)}: " + ", ".join(repr(v) for v in values)
        for (table, column), values in grouped.items()
    )


@lru_cache(maxsize=4)  # stages visit databases in order; big indexes take RAM
def _cached_index(db_path: str, mtime: float, cache_file: str | None) -> ValueIndex:
    del mtime  # part of the cache key only
    if cache_file and Path(cache_file).is_file():
        try:
            return ValueIndex.load(cache_file)
        except (ValueError, pickle.UnpicklingError, EOFError):
            log.warning("rebuilding unreadable value index %s", cache_file)
    index = ValueIndex.build(db_path)
    if cache_file:
        index.save(cache_file)
    return index


def load_value_index(db_path: str | Path, cache_dir: str | Path | None = None) -> ValueIndex:
    """Build or load the value index of ``db_path`` (cached in memory and on disk)."""
    path = Path(db_path).resolve()
    mtime = path.stat().st_mtime
    cache_file = None
    if cache_dir is not None:
        cache_file = str(Path(cache_dir) / "values" / f"{path.stem}-{int(mtime)}-v{INDEX_VERSION}.pkl")
    return _cached_index(str(path), mtime, cache_file)
