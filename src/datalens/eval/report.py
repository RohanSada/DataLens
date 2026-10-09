"""Cross-run report: the small-vs-large comparison tables and figures.

``build_report(run_dirs, out_dir)`` reads each run's ``config.yaml`` and
``scored.jsonl`` and writes:

* ``results.md``: headline table, paired significance tests, self-consistency
  scaling, error taxonomy and cost, ready to paste into the README.
* ``results.json``: the same numbers, machine-readable.
* ``figures/*.png``: light and ``-dark`` variants of each chart.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datalens.data.benchmarks import benchmark_label
from datalens.eval.config import EvalConfig
from datalens.eval.generate import CONFIG_FILE
from datalens.eval.metrics import load_scored, per_question_correct, summarize
from datalens.eval.score import CATEGORIES
from datalens.eval.stats import PairedComparison, holm_adjust, paired_comparison


@dataclass
class Run:
    path: Path
    config: EvalConfig
    scored: list[dict[str, Any]]
    summary: dict[str, Any]

    @property
    def name(self) -> str:
        return self.config.model.name

    @property
    def group(self) -> str:
        return self.config.model.group

    def ex(self) -> float | None:
        greedy = self.summary.get("greedy")
        return greedy["ex"] if greedy else None


def load_runs(run_dirs: Sequence[str | Path], *, n_boot: int = 10_000) -> list[Run]:
    runs = []
    for d in run_dirs:
        path = Path(d)
        config = EvalConfig.from_yaml(path / CONFIG_FILE)
        scored = load_scored(path)
        runs.append(Run(path, config, scored, summarize(scored, config, n_boot=n_boot)))
    return runs


def _pct(value: float | None, digits: int = 1) -> str:
    return (
        "–"
        if value is None or (isinstance(value, float) and math.isnan(value))
        else f"{100 * value:.{digits}f}"
    )


def _usd(value: float | None) -> str:
    if value is None:
        return "–"
    return f"${value:,.2f}" if value >= 0.1 else f"${value:.3f}"


def _seconds(value: float | None) -> str:
    return "–" if value is None else f"{value:.2f}s"


def _p(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


@dataclass
class PairResult:
    small: str
    large: str
    mode: str
    comparison: PairedComparison
    p_holm: float


def paired_tests(runs: list[Run], *, n_boot: int = 10_000) -> list[PairResult]:
    """Every fine-tuned run vs every frontier run on their shared questions.

    The fine-tuned model is compared greedy-vs-greedy, and also with
    self-consistency when it has samples (the deployment setting), against the
    frontier model's greedy answer.
    """
    small = [r for r in runs if r.group == "fine-tuned"]
    large = [r for r in runs if r.group == "frontier"]
    raw: list[tuple[str, str, str, PairedComparison]] = []
    for s in small:
        modes = ["greedy"] + (["sc"] if s.summary.get("sampled") else [])
        large_correct = {lr.name: per_question_correct(lr.scored, "greedy") for lr in large}
        for mode in modes:
            s_correct = per_question_correct(s.scored, mode)
            for lr in large:
                ids = sorted(set(s_correct) & set(large_correct[lr.name]))
                if not ids:
                    continue
                comp = paired_comparison(
                    [s_correct[i] for i in ids], [large_correct[lr.name][i] for i in ids], n_boot=n_boot
                )
                raw.append((s.name, lr.name, mode, comp))
    adjusted = holm_adjust([c.p_value for *_, c in raw])
    return [PairResult(a, b, m, c, p) for (a, b, m, c), p in zip(raw, adjusted, strict=True)]


def _headline_table(runs: list[Run]) -> list[str]:
    lines = [
        "| Model | Group | Params | Training | EX % (95% CI) | Simple | Moderate | Challenging "
        "| Soft-F1 | R-VES | EX % with SC |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(runs, key=lambda r: -(r.ex() or 0)):
        g = r.summary.get("greedy") or {}
        diff = g.get("ex_by_difficulty", {})
        sc = r.summary.get("sampled")
        ci = g.get("ex_ci95")
        ex_cell = f"**{_pct(g.get('ex'))}** ({_pct(ci[0])}–{_pct(ci[1])})" if ci else "–"
        sc_cell = f"{_pct(sc['ex_self_consistency'])} (n={sc['num_samples']})" if sc else "–"
        lines.append(
            f"| {r.name} | {r.group} | {_params(r.config.model.params_b)} | {r.config.model.training} "
            f"| {ex_cell} | {_pct(diff.get('simple', {}).get('ex'))} "
            f"| {_pct(diff.get('moderate', {}).get('ex'))} | {_pct(diff.get('challenging', {}).get('ex'))} "
            f"| {_pct(g.get('soft_f1'))} | {_num(g.get('r_ves'))} "
            f"| {sc_cell} |"
        )
    return lines


def _params(value: float | None) -> str:
    if value is None:
        return "undisclosed"
    return f"{value:g}B"


def _num(value: float | None) -> str:
    return "–" if value is None else f"{value:.1f}"


def _pairs_table(pairs: list[PairResult]) -> list[str]:
    if not pairs:
        return ["No fine-tuned vs frontier pairs in this report."]
    lines = [
        "| Fine-tuned model | vs frontier model | Decoding | Δ EX (pts, 95% CI) | Only small right "
        "| Only large right | McNemar p | Holm p |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for p in pairs:
        c = p.comparison
        mode = "greedy" if p.mode == "greedy" else "self-consistency"
        lines.append(
            f"| {p.small} | {p.large} | {mode} | {100 * c.diff:+.1f} ({100 * c.diff_ci[0]:+.1f} to "
            f"{100 * c.diff_ci[1]:+.1f}) | {c.only_a} | {c.only_b} | {_p(c.p_value)} | {_p(p.p_holm)} |"
        )
    return lines


def _scaling_table(runs: list[Run]) -> list[str]:
    sampled = [r for r in runs if r.summary.get("sampled")]
    if not sampled:
        return ["No run has sampled completions."]
    ks = sorted({int(k) for r in sampled for k in r.summary["sampled"]["maj_at_k"]})
    lines = [
        "| Model | "
        + " | ".join(f"maj@{k}" for k in ks)
        + " | "
        + " | ".join(f"pass@{k}" for k in ks)
        + " |",
        "|---|" + "---:|" * (2 * len(ks)),
    ]
    for r in sampled:
        s = r.summary["sampled"]
        maj = [_pct(s["maj_at_k"].get(str(k))) for k in ks]
        pas = [_pct(s["pass_at_k"].get(str(k))) for k in ks]
        lines.append(f"| {r.name} | " + " | ".join(maj) + " | " + " | ".join(pas) + " |")
    return lines


def _error_table(runs: list[Run]) -> list[str]:
    lines = [
        "| Model | "
        + " | ".join(c.replace("_", " ") for c in CATEGORIES)
        + " | Table recall | Column recall |",
        "|---|" + "---:|" * (len(CATEGORIES) + 2),
    ]
    for r in runs:
        g = r.summary.get("greedy")
        if not g:
            continue
        n = max(sum(g["errors"].values()), 1)
        cells = [_pct(g["errors"][c] / n) for c in CATEGORIES]
        link = g["schema_linking"]
        lines.append(
            f"| {r.name} | "
            + " | ".join(cells)
            + f" | {_pct(link['table_recall'])} | {_pct(link['column_recall'])} |"
        )
    return lines


def _cost_table(runs: list[Run]) -> list[str]:
    lines = [
        "| Model | Decoding | Input tok | Output tok | p50 latency | p95 latency | $ / 1k queries "
        "| $ / 1k correct answers |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in runs:
        rows = [("greedy", r.summary.get("greedy"), "ex")]
        sampled = r.summary.get("sampled")
        if sampled:
            rows.append((f"SC@{sampled['num_samples']}", sampled, "ex_self_consistency"))
        for label, block, ex_key in rows:
            if not block:
                continue
            cost = block["cost"]
            usd_1k = cost["usd_per_1k_queries"]
            ex = block.get(ex_key)
            per_correct = usd_1k / ex if usd_1k is not None and ex else None
            lines.append(
                f"| {r.name} | {label} | {_num(cost['mean_input_tokens'])} "
                f"| {_num(cost['mean_output_tokens'])} "
                f"| {_seconds(cost['latency_p50_s'])} | {_seconds(cost['latency_p95_s'])} | {_usd(usd_1k)} "
                f"| {_usd(per_correct)} |"
            )
    return lines


def write_markdown(runs: list[Run], pairs: list[PairResult], out: Path, figures: list[str]) -> Path:
    benchmarks = sorted({benchmark_label(r.config.benchmark) for r in runs})
    n_questions = sorted({r.summary["n"] for r in runs})
    sections = [
        "# Results",
        "",
        f"Benchmark: {', '.join(benchmarks)} ({', '.join(map(str, n_questions))} questions). "
        "EX is execution accuracy: the share of questions where the predicted query returns the same "
        "set of rows as the gold query. Every model sees the same prompt.",
        "",
        "## Headline",
        "",
        *_headline_table(runs),
        "",
        "## Is the difference real?",
        "",
        "Paired on identical questions. Δ is fine-tuned minus frontier, with a paired-bootstrap 95% CI. "
        "McNemar's exact test uses only the questions where exactly one model is right; Holm-adjusted "
        "p-values correct for testing several pairs at once.",
        "",
        *_pairs_table(pairs),
        "",
        "## Test-time compute: self-consistency",
        "",
        "maj@k: accuracy when k sampled queries vote by execution result. pass@k: share of questions "
        "where at least one of k samples is correct (an upper bound for any reranker).",
        "",
        *_scaling_table(runs),
        "",
        "## Why queries fail (greedy, % of questions)",
        "",
        *_error_table(runs),
        "",
        "## Cost and latency",
        "",
        "API models are priced per token (prompt-cache reads included). Local models are priced by "
        "amortised GPU time per query at the hourly rate in their config.",
        "",
        *_cost_table(runs),
        "",
    ]
    if figures:
        sections += ["## Figures", ""] + [f"![{f}](figures/{f}.png)" for f in figures] + [""]
    path = out / "results.md"
    path.write_text("\n".join(sections), encoding="utf-8")
    return path


def build_report(
    run_dirs: Sequence[str | Path], out_dir: str | Path, *, n_boot: int = 10_000, plots: bool = True
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(run_dirs, n_boot=n_boot)
    if not runs:
        raise ValueError("no runs given")
    pairs = paired_tests(runs, n_boot=n_boot)

    figures: list[str] = []
    if plots:
        from datalens.eval import plots as plot_mod

        figures = plot_mod.render_all(runs, out / "figures")

    payload = {
        "runs": [r.summary for r in runs],
        "pairs": [
            {
                "small": p.small,
                "large": p.large,
                "decoding": p.mode,
                "n": p.comparison.n,
                "diff": p.comparison.diff,
                "diff_ci95": list(p.comparison.diff_ci),
                "only_small": p.comparison.only_a,
                "only_large": p.comparison.only_b,
                "p_value": p.comparison.p_value,
                "p_holm": p.p_holm,
            }
            for p in pairs
        ],
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return write_markdown(runs, pairs, out, figures)
