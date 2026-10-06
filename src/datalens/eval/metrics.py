"""Phase 3 of an evaluation run: aggregate per-question scores into metrics."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from datalens.data.benchmarks import DIFFICULTIES
from datalens.eval.config import EvalConfig, PricingSpec
from datalens.eval.generate import CONFIG_FILE
from datalens.eval.score import CATEGORIES, SCORED_FILE
from datalens.eval.stats import bootstrap_ci, maj_at_k, pass_at_k

METRICS_FILE = "metrics.json"


def load_scored(run_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(run_dir) / SCORED_FILE
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _mean(values: list[float | None]) -> float | None:
    kept = [v for v in values if v is not None]
    return statistics.fmean(kept) if kept else None


def _percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def query_cost_usd(usage: dict[str, Any], pricing: PricingSpec) -> float | None:
    """Dollar cost of one question's generation from token usage or GPU time."""
    if pricing.gpu_hourly_usd is not None:
        latency = usage.get("latency_s")
        if latency is None:
            return None
        return latency * pricing.gpu_hourly_usd * pricing.num_gpus / 3600
    if pricing.input_per_mtok is None or pricing.output_per_mtok is None:
        return None
    total_in = usage.get("input_tokens") or 0
    cached = usage.get("cached_input_tokens") or 0
    written = usage.get("cache_write_tokens") or 0
    out = usage.get("output_tokens") or 0
    read_price = (
        pricing.cache_read_per_mtok if pricing.cache_read_per_mtok is not None else pricing.input_per_mtok
    )
    write_price = (
        pricing.cache_write_per_mtok if pricing.cache_write_per_mtok is not None else pricing.input_per_mtok
    )
    fresh = max(total_in - cached - written, 0)
    return (
        fresh * pricing.input_per_mtok
        + cached * read_price
        + written * write_price
        + out * pricing.output_per_mtok
    ) / 1e6


def _cost_block(usages: list[dict[str, Any]], pricing: PricingSpec) -> dict[str, Any]:
    latencies = [u["latency_s"] for u in usages if u.get("latency_s") is not None]
    costs = [query_cost_usd(u, pricing) for u in usages]
    known = [c for c in costs if c is not None]
    return {
        "mean_input_tokens": _mean([u.get("input_tokens") for u in usages]),
        "mean_output_tokens": _mean([u.get("output_tokens") for u in usages]),
        "latency_p50_s": _percentile(latencies, 50),
        "latency_p95_s": _percentile(latencies, 95),
        "queries_per_s": (1 / statistics.fmean(latencies))
        if latencies and statistics.fmean(latencies) > 0
        else None,
        "usd_per_1k_queries": (1000 * statistics.fmean(known))
        if known and len(known) == len(costs)
        else None,
    }


def summarize(scored: list[dict[str, Any]], config: EvalConfig, *, n_boot: int = 10_000) -> dict[str, Any]:
    """Headline and diagnostic metrics for one run."""
    model = config.model
    summary: dict[str, Any] = {
        "run_name": config.run_name,
        "model": model.name,
        "model_id": model.model,
        "group": model.group,
        "training": model.training,
        "params_b": model.params_b,
        "train_fraction": model.train_fraction,
        "benchmark": config.benchmark,
        "n": len(scored),
        "gold_failures": sum(1 for s in scored if not s["gold_ok"]),
    }

    greedy = [s for s in scored if s.get("greedy")]
    if greedy:
        correct = [float(s["greedy"]["correct"]) for s in greedy]
        lo, hi = bootstrap_ci(correct, n_boot=n_boot)
        by_difficulty = {}
        for level in DIFFICULTIES:
            subset = [float(s["greedy"]["correct"]) for s in greedy if s.get("difficulty") == level]
            if subset:
                by_difficulty[level] = {"n": len(subset), "ex": statistics.fmean(subset)}
        categories = Counter(s["greedy"]["category"] for s in greedy)
        ves = [s["greedy"].get("ves_reward") for s in greedy]
        summary["greedy"] = {
            "ex": statistics.fmean(correct),
            "ex_ci95": [lo, hi],
            "ex_by_difficulty": by_difficulty,
            "soft_f1": statistics.fmean(s["greedy"]["soft_f1"] for s in greedy),
            "r_ves": (100 * statistics.fmean(v for v in ves if v is not None))
            if all(v is not None for v in ves)
            else None,
            "errors": {c: categories.get(c, 0) for c in CATEGORIES},
            "schema_linking": {
                key: _mean([s["greedy"].get(key) for s in greedy])
                for key in ("table_recall", "table_precision", "column_recall", "column_precision")
            },
            "truncated": sum(1 for s in greedy if s["greedy"].get("finish_reason") == "length"),
            "cost": _cost_block([s["greedy"] for s in greedy], config.pricing),
        }

    sampled = [s for s in scored if s.get("samples") and s["samples"]["keys"]]
    if sampled:
        n = min(len(s["samples"]["keys"]) for s in sampled)
        ks = sorted({k for k in (1, 2, 4, 8, 16, 32, 64) if k <= n} | {n})
        maj = {}
        passk = {}
        for k in ks:
            maj_scores = [
                maj_at_k(
                    s["samples"]["keys"][:n],
                    s["samples"]["correct"][:n],
                    s["samples"]["empty"][:n],
                    k,
                    seed=i,
                )
                for i, s in enumerate(sampled)
            ]
            pass_scores = [pass_at_k(n, sum(s["samples"]["correct"][:n]), k) for s in sampled]
            maj[str(k)] = statistics.fmean(maj_scores)
            passk[str(k)] = statistics.fmean(pass_scores)
        # Self-consistency over all samples is the "+SC" headline number.
        full = [
            maj_at_k(s["samples"]["keys"][:n], s["samples"]["correct"][:n], s["samples"]["empty"][:n], n)
            for s in sampled
        ]
        lo, hi = bootstrap_ci(full, n_boot=n_boot)
        summary["sampled"] = {
            "num_samples": n,
            "maj_at_k": maj,
            "pass_at_k": passk,
            "ex_self_consistency": statistics.fmean(full),
            "ex_self_consistency_ci95": [lo, hi],
            "cost": _cost_block([s["samples"] for s in sampled], config.pricing),
        }
    return summary


def write_metrics(run_dir: str | Path, *, n_boot: int = 10_000) -> dict[str, Any]:
    run_dir = Path(run_dir)
    config = EvalConfig.from_yaml(run_dir / CONFIG_FILE)
    summary = summarize(load_scored(run_dir), config, n_boot=n_boot)
    (run_dir / METRICS_FILE).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def per_question_correct(scored: list[dict[str, Any]], mode: str = "greedy") -> dict[str, bool]:
    """Map question id to correctness, for paired comparisons across runs.

    ``mode="greedy"`` uses the greedy prediction; ``mode="sc"`` uses majority
    voting over all samples.
    """
    out: dict[str, bool] = {}
    for s in scored:
        if mode == "greedy" and s.get("greedy"):
            out[s["id"]] = bool(s["greedy"]["correct"])
        elif mode == "sc" and s.get("samples") and s["samples"]["keys"]:
            smp = s["samples"]
            out[s["id"]] = maj_at_k(smp["keys"], smp["correct"], smp["empty"], len(smp["keys"])) == 1.0
    return out
