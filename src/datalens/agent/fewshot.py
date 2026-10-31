"""Worked examples: similar solved questions from BIRD train, shown in the prompt.

BIRD's reference queries follow conventions a model can't infer from the schema
(how percentages are computed, when a ratio is cast to REAL, that "the highest"
means ``ORDER BY ... LIMIT 1``). A few similar train questions with their gold
SQL show those conventions. Retrieval is TF-IDF cosine over question and
evidence text, and examples from the question's own database are never used,
so a tuning run on train sees the same conditions as a dev run (whose databases
are not in train at all).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from datalens.data.benchmarks import Example

_WORD_RE = re.compile(r"[a-z]+|\d+")


def _terms(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


@dataclass(frozen=True)
class Shot:
    question: str
    evidence: str
    sql: str
    db_id: str
    score: float


class FewShotIndex:
    def __init__(self, examples: Sequence[Example]) -> None:
        self.examples = [e for e in examples if e.gold_sql]
        docs = [Counter(_terms(f"{e.question} {e.evidence}")) for e in self.examples]
        df: Counter[str] = Counter()
        for doc in docs:
            df.update(doc.keys())
        n = max(len(docs), 1)
        self.idf = {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}
        self.vectors = [self._weigh(doc) for doc in docs]
        self.postings: dict[str, list[int]] = {}
        for i, vec in enumerate(self.vectors):
            for t in vec:
                self.postings.setdefault(t, []).append(i)

    def _weigh(self, counts: Counter[str]) -> dict[str, float]:
        vec = {t: (1 + math.log(c)) * self.idf.get(t, 0.0) for t, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items() if v > 0}

    def query(self, question: str, evidence: str, *, k: int = 3, exclude_db: str | None = None) -> list[Shot]:
        """The ``k`` most similar examples from other databases, without near-duplicate SQL."""
        if k <= 0 or not self.examples:
            return []
        q = self._weigh(Counter(_terms(f"{question} {evidence}")))
        scores: dict[int, float] = {}
        for t, w in q.items():
            for i in self.postings.get(t, []):
                scores[i] = scores.get(i, 0.0) + w * self.vectors[i][t]
        shots: list[Shot] = []
        seen_sql: set[str] = set()
        for i, score in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0])):
            e = self.examples[i]
            if e.db_id == exclude_db or e.question.strip() == question.strip():
                continue
            key = " ".join(e.gold_sql.lower().split())
            if key in seen_sql:
                continue
            seen_sql.add(key)
            shots.append(Shot(e.question, e.evidence, e.gold_sql.strip(), e.db_id, score))
            if len(shots) >= k:
                break
        return shots


def format_shots(shots: Sequence[Shot]) -> str:
    if not shots:
        return ""
    lines = [
        "Solved examples from other databases (their tables differ; use them for the style of the "
        "reference queries, such as how ratios, percentages and rankings are written):"
    ]
    for s in shots:
        lines += ["", f"Question: {s.question}"]
        if s.evidence:
            lines.append(f"External knowledge: {s.evidence}")
        lines.append(f"SQL: {' '.join(s.sql.split())}")
    return "\n".join(lines)
