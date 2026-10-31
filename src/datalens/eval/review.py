"""A per-question review file for error analysis away from the GPU machine.

``review.jsonl`` has one row per question: the question, the gold and chosen
queries, the error category, and for wrong answers a preview of both results.
For agent runs it also records which attempts were right and how the answer was
chosen, which separates "no attempt found it" (a generation problem) from "an
attempt found it but wasn't picked" (a selection problem).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datalens.agent.prompts import format_rows
from datalens.data.benchmarks import Example
from datalens.eval.config import EvalConfig
from datalens.eval.generate import CONFIG_FILE, GENERATIONS_FILE, load_benchmark
from datalens.eval.metrics import load_scored
from datalens.sql.executor import ExecResult, execute

REVIEW_FILE = "review.jsonl"


def _preview(result: ExecResult, rows: int) -> str:
    if not result.ok:
        return f"Error: {result.error}"
    return format_rows(result.columns, result.rows or [], truncated=result.truncated, max_rows=rows)


def write_review(
    run_dir: str | Path,
    *,
    examples: list[Example] | None = None,
    preview_rows: int = 5,
    timeout_s: float = 30.0,
) -> Path:
    run_dir = Path(run_dir)
    config = EvalConfig.from_yaml(run_dir / CONFIG_FILE)
    examples = examples if examples is not None else load_benchmark(config)
    by_id = {e.id: e for e in examples}
    generations: dict[str, dict[str, Any]] = {}
    for line in (run_dir / GENERATIONS_FILE).read_text(encoding="utf-8").splitlines():
        if line.strip():
            generation = json.loads(line)
            generations[generation["id"]] = generation

    out_path = run_dir / REVIEW_FILE
    with out_path.open("w", encoding="utf-8") as fh:
        for scored in load_scored(run_dir):
            example = by_id.get(scored["id"])
            greedy = scored.get("greedy")
            if example is None or greedy is None:
                continue
            row: dict[str, Any] = {
                "id": example.id,
                "db_id": example.db_id,
                "difficulty": example.difficulty,
                "question": example.question,
                "evidence": example.evidence,
                "gold_sql": example.gold_sql,
                "pred_sql": greedy["sql"],
                "category": greedy["category"],
                "correct": greedy["correct"],
            }
            samples = scored.get("samples")
            if samples:
                row["samples_correct"] = samples["correct"]
            agent = generations.get(example.id, {}).get("agent")
            if agent:
                correct = samples["correct"] if samples else []
                row["agent"] = {
                    "method": agent["method"],
                    "chosen_group": agent["chosen_group"],
                    "groups": [
                        {
                            "members": g["members"],
                            "rows": g["rows"],
                            "correct": bool(correct and correct[g["members"][0]]),
                            "sql": g["sql"],
                        }
                        for g in agent["groups"]
                    ],
                    "scores": agent.get("scores"),
                    "statuses": [a["status"] for a in agent["attempts"]],
                    "turns": [a["turns"] for a in agent["attempts"]],
                }
            if not greedy["correct"]:
                gold = execute(example.db_path, example.gold_sql, timeout_s=timeout_s, max_rows=200)
                row["gold_result"] = _preview(gold, preview_rows)
                if greedy["sql"]:
                    pred = execute(example.db_path, greedy["sql"], timeout_s=timeout_s, max_rows=200)
                    row["pred_result"] = _preview(pred, preview_rows)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out_path
