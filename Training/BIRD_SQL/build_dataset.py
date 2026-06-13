"""Build JSONL training data from a BIRD-format question file.

Example
-------
Full-schema (recommended for BIRD; no Qdrant/Neo4j needed)::

    python -m Training.BIRD_SQL.build_dataset \
        --input Data/BIRD_SQL/minidev/MINIDEV/mini_dev_sqlite.json \
        --out-dir Training/BIRD_SQL/out \
        --schema-strategy full \
        --report-coverage

Retrieval-augmented, with a held-out validation split::

    python -m Training.BIRD_SQL.build_dataset \
        --input <bird.json> --out-dir <dir> \
        --schema-strategy gold_union --num-distractors 2 \
        --val-split 0.05

Output: ``train.jsonl`` (and ``val.jsonl`` when ``--val-split`` > 0). Each line
is ``{prompt, completion, db_id, question_id, difficulty}``.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

# schema_retrieval flat config (made importable by this package's __init__).
from config import get_settings, load_settings

from .config import BuildConfig
from .gold_sql import extract_referenced_tables
from .prompt_template import build_example
from .schema_context import build_provider, load_schema_map

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def _load_examples(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON list of examples in {path}, got {type(data).__name__}"
        )
    return data


def _write_jsonl(path: Path, records: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Wrote %d records to %s", len(records), path)


def _resolve_settings(cfg: BuildConfig):
    if cfg.settings_path is not None:
        return load_settings(cfg.settings_path)
    return get_settings()


# ---------------------------------------------------------------------------
# Coverage report
# ---------------------------------------------------------------------------
def _schema_contains_table(context: str, table: str) -> bool:
    # Matches the renderer output: ``CREATE TABLE <name> (``.
    return f"CREATE TABLE {table} (" in context


class _Coverage:
    """Tracks gold-table coverage of produced contexts, overall + sliced."""

    def __init__(self) -> None:
        self.total = 0
        self.covered = 0
        self.by_difficulty: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
        self.by_db: Dict[str, List[int]] = defaultdict(lambda: [0, 0])

    def update(self, gold_tables, context: str, difficulty: str, db_id: str) -> None:
        if not gold_tables:
            return
        ok = all(_schema_contains_table(context, t) for t in gold_tables)
        self.total += 1
        self.covered += int(ok)
        self.by_difficulty[difficulty][0] += int(ok)
        self.by_difficulty[difficulty][1] += 1
        self.by_db[db_id][0] += int(ok)
        self.by_db[db_id][1] += 1

    def report(self) -> str:
        if self.total == 0:
            return "Coverage: no parseable gold SQL to evaluate."
        lines = [
            "Schema-linking coverage (gold tables present in context):",
            f"  overall: {self.covered}/{self.total} "
            f"({100.0 * self.covered / self.total:.1f}%)",
            "  by difficulty:",
        ]
        for diff, (ok, tot) in sorted(self.by_difficulty.items()):
            pct = 100.0 * ok / tot if tot else 0.0
            lines.append(f"    {diff:12s}: {ok}/{tot} ({pct:.1f}%)")
        lines.append("  by database:")
        for db, (ok, tot) in sorted(self.by_db.items()):
            pct = 100.0 * ok / tot if tot else 0.0
            lines.append(f"    {db:28s}: {ok}/{tot} ({pct:.1f}%)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core build
# ---------------------------------------------------------------------------
def build(cfg: BuildConfig) -> dict:
    settings = _resolve_settings(cfg)
    schemas = load_schema_map(settings)
    logger.info("Loaded %d database schemas", len(schemas))

    examples = _load_examples(cfg.input_path)
    if cfg.db_filter:
        examples = [e for e in examples if e.get("db_id") == cfg.db_filter]
    if cfg.limit is not None:
        examples = examples[: cfg.limit]
    logger.info("Processing %d examples (strategy=%s)", len(examples), cfg.schema_strategy)

    provider = build_provider(cfg, schemas, settings)
    coverage = _Coverage() if cfg.report_coverage else None

    records: List[dict] = []
    skipped = 0
    try:
        for i, ex in enumerate(examples):
            db_id = ex.get("db_id")
            question = ex.get("question", "")
            evidence = ex.get("evidence", "") or ""
            sql = ex.get("SQL", ex.get("sql", "")) or ""
            try:
                context = provider.context_for(db_id, question, evidence, sql)
            except Exception as exc:  # noqa: BLE001 - keep long runs alive
                logger.warning(
                    "Skipping example %s (db '%s'): %s",
                    ex.get("question_id", i),
                    db_id,
                    exc,
                )
                skipped += 1
                continue

            body = build_example(context, question, evidence, sql)
            body.update(
                {
                    "db_id": db_id,
                    "question_id": ex.get("question_id"),
                    "difficulty": ex.get("difficulty"),
                }
            )
            records.append(body)

            if coverage is not None and sql:
                schema = schemas.get(db_id)
                known = [t.name for t in schema.tables] if schema else []
                gold_tables = extract_referenced_tables(sql, known)
                coverage.update(
                    gold_tables, context, ex.get("difficulty", "?"), db_id
                )

            if (i + 1) % 100 == 0:
                logger.info("  ...processed %d/%d", i + 1, len(examples))
    finally:
        provider.close()

    _emit(cfg, records)

    summary = {
        "input": str(cfg.input_path),
        "out_dir": str(cfg.out_dir),
        "strategy": cfg.schema_strategy,
        "written": len(records),
        "skipped": skipped,
    }
    if coverage is not None:
        print(coverage.report())
        summary["coverage_overall"] = (
            coverage.covered / coverage.total if coverage.total else None
        )
    logger.info("Build summary: %s", summary)
    return summary


def _emit(cfg: BuildConfig, records: List[dict]) -> None:
    if cfg.val_split and cfg.val_split > 0.0:
        rng = random.Random(cfg.seed)
        shuffled = records[:]
        rng.shuffle(shuffled)
        n_val = int(round(len(shuffled) * cfg.val_split))
        val = shuffled[:n_val]
        train = shuffled[n_val:]
        _write_jsonl(cfg.out_dir / "train.jsonl", train)
        _write_jsonl(cfg.out_dir / "val.jsonl", val)
    else:
        _write_jsonl(cfg.out_dir / "train.jsonl", records)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> BuildConfig:
    parser = argparse.ArgumentParser(
        description="Build JSONL {prompt, completion} training data from a "
        "BIRD-format question file."
    )
    parser.add_argument("--input", required=True, type=Path, help="BIRD-format JSON file")
    parser.add_argument("--out-dir", required=True, type=Path, help="output directory")
    parser.add_argument(
        "--schema-strategy",
        choices=["full", "retrieved", "gold_union"],
        default="full",
    )
    parser.add_argument(
        "--include-descriptions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include column descriptions as inline comments",
    )
    parser.add_argument(
        "--include-value-hints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include value descriptions / sampled example values",
    )
    parser.add_argument(
        "--include-fk",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include foreign-key join paths",
    )
    parser.add_argument(
        "--num-distractors",
        type=int,
        default=0,
        help="gold_union only: extra random tables added to the context",
    )
    parser.add_argument(
        "--schema-token-budget",
        type=int,
        default=None,
        help="approx token cap on the rendered schema (default: no cap)",
    )
    parser.add_argument("--val-split", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--db", dest="db_filter", default=None, help="filter to one db_id")
    parser.add_argument(
        "--settings",
        dest="settings_path",
        type=Path,
        default=None,
        help="path to schema_retrieval settings.json (default: package default)",
    )
    parser.add_argument("--report-coverage", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="cap #examples (debug)")
    args = parser.parse_args(argv)

    return BuildConfig(
        input_path=args.input,
        out_dir=args.out_dir,
        schema_strategy=args.schema_strategy,
        include_descriptions=args.include_descriptions,
        include_value_hints=args.include_value_hints,
        include_fk=args.include_fk,
        num_distractors=args.num_distractors,
        schema_token_budget=args.schema_token_budget,
        val_split=args.val_split,
        seed=args.seed,
        db_filter=args.db_filter,
        settings_path=args.settings_path,
        report_coverage=args.report_coverage,
        limit=args.limit,
    )


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    cfg = _parse_args(argv)
    build(cfg)


if __name__ == "__main__":
    main()
