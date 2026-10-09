"""Phase 2 of an evaluation run: execute and grade completions (CPU only).

For each question this records whether the greedy query is correct and *why* it
failed if not, its Soft-F1, how well it linked the schema, and for sampled
completions the result fingerprint and correctness of each sample, which is
everything the self-consistency and pass@k metrics need.
"""

from __future__ import annotations

import json
import logging
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from datalens.data.benchmarks import Example
from datalens.eval.config import EvalConfig
from datalens.eval.generate import CONFIG_FILE, GENERATIONS_FILE, load_benchmark
from datalens.sql.compare import exec_match, result_key, soft_f1, ves_reward
from datalens.sql.executor import ExecResult, execute
from datalens.sql.parse import extract_sql, schema_refs

log = logging.getLogger(__name__)

SCORED_FILE = "scored.jsonl"

CATEGORIES = (
    "correct",
    "no_sql",
    "syntax_error",
    "schema_error",
    "timeout",
    "runtime_error",
    "empty_result",
    "wrong_result",
)


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _schema_link(pred_sql: str | None, gold_sql: str) -> dict[str, float | None]:
    empty: dict[str, float | None] = {
        "table_recall": None,
        "table_precision": None,
        "column_recall": None,
        "column_precision": None,
    }
    gold = schema_refs(gold_sql)
    pred = schema_refs(pred_sql) if pred_sql else None
    if gold is None:
        return empty
    if pred is None:
        # Nothing usable predicted: no gold tables or columns were recovered.
        return {
            "table_recall": 0.0 if gold.tables else None,
            "table_precision": None,
            "column_recall": 0.0 if gold.columns else None,
            "column_precision": None,
        }
    return {
        "table_recall": _ratio(len(pred.tables & gold.tables), len(gold.tables)),
        "table_precision": _ratio(len(pred.tables & gold.tables), len(pred.tables)),
        "column_recall": _ratio(len(pred.columns & gold.columns), len(gold.columns)),
        "column_precision": _ratio(len(pred.columns & gold.columns), len(pred.columns)),
    }


def categorize(sql: str | None, pred: ExecResult | None, gold: ExecResult) -> str:
    """Assign one error-taxonomy category to a prediction."""
    if sql is None or pred is None:
        return "no_sql"
    if not pred.ok:
        assert pred.error_kind is not None
        return pred.error_kind.value
    if gold.ok and gold.rows is not None and pred.rows is not None and exec_match(pred.rows, gold.rows):
        return "correct"
    if not pred.rows and gold.ok and gold.rows:
        return "empty_result"
    return "wrong_result"


class Scorer:
    def __init__(self, timeout_s: float = 30.0, gold_timeout_s: float = 120.0) -> None:
        self.timeout_s = timeout_s
        self.gold_timeout_s = gold_timeout_s

    def score_example(self, example: Example, row: dict[str, Any]) -> dict[str, Any]:
        db = example.db_path
        gold = execute(db, example.gold_sql, timeout_s=self.gold_timeout_s)
        cache: dict[str, ExecResult] = {}

        def run(sql: str) -> ExecResult:
            if sql not in cache:
                cache[sql] = execute(db, sql, timeout_s=self.timeout_s)
            return cache[sql]

        out: dict[str, Any] = {
            "id": example.id,
            "db_id": example.db_id,
            "difficulty": example.difficulty,
            "gold_ok": gold.ok,
            "gold_error": gold.error,
        }

        greedy = row.get("greedy")
        if greedy is not None:
            text = greedy["completions"][0]["text"] if greedy["completions"] else ""
            sql = extract_sql(text)
            pred = run(sql) if sql else None
            category = categorize(sql, pred, gold)
            f1 = 0.0
            if pred is not None and pred.ok and gold.ok and pred.rows is not None and gold.rows is not None:
                f1 = soft_f1(pred.rows, gold.rows)
            out["greedy"] = {
                "sql": sql,
                "category": category,
                "correct": category == "correct",
                "error": pred.error if pred is not None else None,
                "soft_f1": f1,
                "pred_time_s": pred.elapsed_s if pred is not None and pred.ok else None,
                "gold_time_s": gold.elapsed_s if gold.ok else None,
                "finish_reason": greedy["completions"][0].get("finish_reason")
                if greedy["completions"]
                else None,
                **_schema_link(sql, example.gold_sql),
                **_usage(greedy),
            }

        samples = row.get("samples")
        if samples is not None:
            keys: list[str | None] = []
            correct: list[bool] = []
            empty: list[bool] = []
            for completion in samples["completions"]:
                sql = extract_sql(completion["text"])
                res = run(sql) if sql else None
                ok = res is not None and res.ok and res.rows is not None
                keys.append(result_key(res.rows) if ok and res is not None and res.rows is not None else None)
                empty.append(bool(ok and res is not None and not res.rows))
                correct.append(
                    bool(
                        ok
                        and gold.ok
                        and res is not None
                        and res.rows is not None
                        and gold.rows is not None
                        and exec_match(res.rows, gold.rows)
                    )
                )
            out["samples"] = {"keys": keys, "correct": correct, "empty": empty, **_usage(samples)}
        return out


def _usage(record: dict[str, Any]) -> dict[str, Any]:
    completions = record.get("completions") or []
    output_tokens = [c.get("output_tokens") for c in completions]
    return {
        "input_tokens": record.get("input_tokens"),
        "cached_input_tokens": record.get("cached_input_tokens") or 0,
        "cache_write_tokens": record.get("cache_write_tokens") or 0,
        "output_tokens": None if any(t is None for t in output_tokens) else sum(output_tokens),
        "latency_s": record.get("latency_s"),
    }


def _robust_mean(values: list[float]) -> float:
    """Mean after dropping points beyond 3 standard deviations (BIRD's VES recipe)."""
    if len(values) < 3:
        return statistics.fmean(values)
    mu, sd = statistics.fmean(values), statistics.pstdev(values)
    kept = [v for v in values if abs(v - mu) <= 3 * sd] or values
    return statistics.fmean(kept)


def measure_ves(example: Example, pred_sql: str, iterations: int, timeout_s: float) -> float:
    """R-VES reward for a correct prediction, timing gold and prediction alternately."""
    pred_times, gold_times = [], []
    for _ in range(iterations):
        pred = execute(example.db_path, pred_sql, timeout_s=timeout_s)
        gold = execute(example.db_path, example.gold_sql, timeout_s=timeout_s)
        if not (pred.ok and gold.ok):
            return 0.0
        pred_times.append(pred.elapsed_s)
        gold_times.append(gold.elapsed_s)
    pred_mean = _robust_mean(pred_times)
    return ves_reward(_robust_mean(gold_times) / pred_mean if pred_mean > 0 else 0.0)


def score_run(
    run_dir: str | Path,
    *,
    examples: list[Example] | None = None,
    workers: int = 8,
    timeout_s: float = 30.0,
    ves_iterations: int = 0,
) -> Path:
    """Score ``generations.jsonl`` in ``run_dir`` and write ``scored.jsonl``.

    ``ves_iterations > 0`` additionally measures R-VES for correct greedy
    predictions. Timing runs sequentially after the parallel pass so that
    concurrent queries don't distort the measurements.
    """
    run_dir = Path(run_dir)
    config = EvalConfig.from_yaml(run_dir / CONFIG_FILE)
    examples = examples if examples is not None else load_benchmark(config)
    by_id = {e.id: e for e in examples}

    rows = [
        json.loads(line)
        for line in (run_dir / GENERATIONS_FILE).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [r for r in rows if r["id"] in by_id]
    scorer = Scorer(timeout_s=timeout_s)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        scored = list(pool.map(lambda r: scorer.score_example(by_id[r["id"]], r), rows))

    if ves_iterations > 0:
        for item in scored:
            greedy = item.get("greedy")
            if greedy and greedy["correct"]:
                greedy["ves_reward"] = measure_ves(
                    by_id[item["id"]], greedy["sql"], ves_iterations, timeout_s
                )
            elif greedy:
                greedy["ves_reward"] = 0.0

    out_path = run_dir / SCORED_FILE
    with out_path.open("w", encoding="utf-8") as fh:
        for item in sorted(scored, key=lambda s: s["id"]):
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    log.info("scored %d examples -> %s", len(scored), out_path)
    return out_path
