"""Loading BIRD and Spider into one :class:`Example` type."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DIFFICULTIES = ("simple", "moderate", "challenging")


@dataclass(frozen=True)
class Example:
    """One text-to-SQL question with its gold query and database."""

    id: str
    db_id: str
    question: str
    evidence: str
    gold_sql: str
    db_path: Path
    difficulty: str | None = None


@dataclass(frozen=True)
class BenchmarkSpec:
    """Where a benchmark split lives on disk."""

    name: str
    questions: Path
    db_root: Path


def bird_spec(root: str | Path, split: str) -> BenchmarkSpec:
    """BIRD layout produced by ``datalens data download``.

    ``root/<split>/<split>.json`` and ``root/<split>/<split>_databases/<db>/<db>.sqlite``.
    """
    base = Path(root) / split
    return BenchmarkSpec(
        name=f"bird-{split}",
        questions=base / f"{split}.json",
        db_root=base / f"{split}_databases",
    )


def spider_spec(root: str | Path, split: str = "dev") -> BenchmarkSpec:
    """Spider 1.0 layout: ``root/<split>.json`` and ``root/database/<db>/<db>.sqlite``."""
    base = Path(root)
    return BenchmarkSpec(name=f"spider-{split}", questions=base / f"{split}.json", db_root=base / "database")


def resolve_spec(benchmark: str, root: str | Path) -> BenchmarkSpec:
    """Parse names like ``bird-dev``, ``bird-train`` or ``spider-dev``."""
    family, _, split = benchmark.partition("-")
    if family == "bird" and split:
        return bird_spec(root, split)
    if family == "spider":
        return spider_spec(root, split or "dev")
    raise ValueError(f"unknown benchmark {benchmark!r}; expected bird-<split> or spider-<split>")


def benchmark_label(benchmark: str) -> str:
    """Display name for reports and figures, e.g. ``spider-dev`` -> ``Spider dev``."""
    family, _, split = benchmark.partition("-")
    name = {"bird": "BIRD", "spider": "Spider"}.get(family, family)
    return f"{name} {split}".strip()


def load_examples(spec: BenchmarkSpec, *, limit: int | None = None) -> list[Example]:
    """Load questions from a BIRD- or Spider-format JSON file.

    BIRD rows carry ``SQL``, ``evidence``, ``difficulty`` and (dev only)
    ``question_id``; Spider rows carry ``query`` and nothing else.
    """
    if not spec.questions.is_file():
        raise FileNotFoundError(
            f"{spec.questions} not found. Run `datalens data download` or pass --data-root."
        )
    rows = json.loads(spec.questions.read_text(encoding="utf-8"))
    examples = []
    for idx, row in enumerate(rows):
        db_id = row["db_id"]
        qid = row.get("question_id", idx)
        examples.append(
            Example(
                id=f"{spec.name}-{qid}",
                db_id=db_id,
                question=row["question"].strip(),
                evidence=(row.get("evidence") or "").strip(),
                gold_sql=(row.get("SQL") or row.get("query") or "").strip(),
                db_path=spec.db_root / db_id / f"{db_id}.sqlite",
                difficulty=row.get("difficulty"),
            )
        )
        if limit is not None and len(examples) >= limit:
            break
    return examples
