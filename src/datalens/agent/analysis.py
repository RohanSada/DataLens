"""The paper's analyses, computed from one finished agent run.

* **Ablations** (paper Table 2): drop the reasoning generator, the ICL
  generator, refinement, or the tournament (falling back to self-consistency).
  They are computed from the full run: the tournament is re-scored using only
  the candidates and judgments that remain, rather than rerunning the pipeline.
  The task-understanding ablation changes the prompts, so it needs its own run
  (``understanding.enabled: false``); pass it with ``compare``.
* **Generators** (Fig. 3-5): voting accuracy, selection accuracy and upper bound
  per generator, and how many questions only one of them solves.
* **Candidates** (Fig. 6): pass@k as the number of generated candidates grows.
* **Selector**: accuracy on pairs where exactly one query is right, and how
  often it picks the first position.
* **Cost** of each stage.

Correctness compares result fingerprints with the gold query's, which matches
BIRD's set-equality rule; the final answer is cross-checked against
``scored.jsonl`` when the run has been scored.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datalens.agent.execution import QueryRunner
from datalens.agent.pipeline import STAGE_FILES, STAGES, AgentRun, load_agent_config, read_jsonl
from datalens.agent.selection import Candidate, Selection, restricted_choice
from datalens.agent.stages import Item, Usage
from datalens.data.benchmarks import DIFFICULTIES, Example
from datalens.eval.metrics import load_scored, per_question_correct, query_cost_usd
from datalens.eval.score import SCORED_FILE
from datalens.eval.stats import paired_comparison, pass_at_k

ANALYSIS_FILE = "analysis.json"
ANALYSIS_MD = "analysis.md"


@dataclass
class Question:
    example: Example
    pool: list[Candidate]
    correct: list[bool]
    selection: Selection | None
    gold_ok: bool

    def ids(self, keep: Callable[[Candidate], bool]) -> set[int]:
        return {i for i, c in enumerate(self.pool) if keep(c)}

    def pick(self, allowed: set[int], *, vote: bool = False) -> int | None:
        if self.selection is None:
            return None
        tournament, majority = restricted_choice(self.pool, self.selection, allowed)
        return majority if vote else tournament

    def right(self, index: int | None) -> bool:
        return index is not None and self.correct[index]


def _is_original(c: Candidate) -> bool:
    return c.generator in ("reasoning", "icl")


VARIANTS: dict[str, tuple[Callable[[Candidate], bool], bool]] = {
    # name: (which candidates stay, select by vote instead of tournament)
    "Agentar-Scale (full)": (lambda c: True, False),
    "w/o reasoning generator": (lambda c: c.root == "icl", False),
    "w/o ICL generator": (lambda c: c.root == "reasoning", False),
    "w/o iterative refinement": (_is_original, False),
    "w/o selection scaling (self-consistency)": (lambda c: True, True),
}


def load_questions(run_dir: str | Path, examples: list[Example] | None = None) -> list[Question]:
    """Pool, per-candidate correctness and selection for every question that finished."""
    run_dir = Path(run_dir)
    config = load_agent_config(run_dir)
    run = AgentRun(config, run_dir.parent, examples=examples)
    state = run.load_state()
    finished = [e for e in run.examples() if e.id in state["select"]]
    runner = QueryRunner(timeout_s=120.0, workers=config.execution.workers)
    gold = runner.run_many([(e.db_path, e.gold_sql) for e in finished])
    questions = []
    for example, gold_outcome in zip(finished, gold, strict=True):
        item: Item = run.build_item(example, state)
        pool = item.pool()
        correct = [bool(gold_outcome.ok and c.outcome.ok and c.outcome.key == gold_outcome.key) for c in pool]
        questions.append(Question(example, pool, correct, item.selection, gold_outcome.ok))
    return questions


def _by_difficulty(questions: Sequence[Question], hit: Callable[[Question], float]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for level in DIFFICULTIES:
        subset = [q for q in questions if q.example.difficulty == level]
        if subset:
            out[level] = statistics.fmean(hit(q) for q in subset)
    out["total"] = statistics.fmean(hit(q) for q in questions) if questions else None
    return out


def _picks_right(keep: Callable[[Candidate], bool], *, vote: bool) -> Callable[[Question], bool]:
    def hit(q: Question) -> bool:
        return q.right(q.pick(q.ids(keep), vote=vote))

    return hit


def _any_right(keep: Callable[[Candidate], bool]) -> Callable[[Question], bool]:
    def hit(q: Question) -> bool:
        return any(q.correct[i] for i in q.ids(keep))

    return hit


def ablations(questions: Sequence[Question]) -> dict[str, dict[str, Any]]:
    return {
        name: _by_difficulty(questions, _picks_right(keep, vote=vote))
        for name, (keep, vote) in VARIANTS.items()
    }


def generators(questions: Sequence[Question]) -> dict[str, dict[str, Any]]:
    groups: dict[str, Callable[[Candidate], bool]] = {
        "ICL generator": lambda c: c.root == "icl",
        "Reasoning generator": lambda c: c.root == "reasoning",
        "All": lambda c: True,
    }
    out = {}
    for name, keep in groups.items():
        out[name] = {
            "voting": _by_difficulty(questions, _picks_right(keep, vote=True)),
            "selection": _by_difficulty(questions, _picks_right(keep, vote=False)),
            "upper_bound": _by_difficulty(questions, _any_right(keep)),
        }
    return out


def overlap(questions: Sequence[Question]) -> dict[str, int]:
    """Questions solvable (some candidate right) by each generator's lineage."""
    both = icl_only = reasoning_only = neither = 0
    for q in questions:
        r = any(q.correct[i] for i in q.ids(lambda c: c.root == "reasoning"))
        i = any(q.correct[j] for j in q.ids(lambda c: c.root == "icl"))
        both += r and i
        reasoning_only += r and not i
        icl_only += i and not r
        neither += not (r or i)
    return {"both": both, "icl_only": icl_only, "reasoning_only": reasoning_only, "neither": neither}


def candidate_scaling(
    questions: Sequence[Question], ks: Sequence[int] = (1, 2, 4, 8, 16, 32)
) -> dict[str, Any]:
    """pass@k over the generated (unrefined) candidates, by difficulty."""
    out: dict[str, Any] = {}
    sizes = [sum(_is_original(c) for c in q.pool) for q in questions]
    largest = max(sizes, default=0)
    for k in [k for k in ks if k <= largest] + ([largest] if largest and largest not in ks else []):

        def hit(q: Question, k: int = k) -> float:
            idx = q.ids(_is_original)
            n, c = len(idx), sum(q.correct[i] for i in idx)
            return pass_at_k(n, c, min(k, n)) if n else 0.0

        out[str(k)] = _by_difficulty(questions, hit)
    return out


def selector_stats(questions: Sequence[Question]) -> dict[str, Any]:
    judged = decisive = right = first = unparsed = 0
    for q in questions:
        sel = q.selection
        if sel is None or not sel.matches:
            continue
        rep = {g.key: g.representative for g in sel.groups}
        for match, winner in zip(sel.matches, sel.winners, strict=True):
            judged += 1
            if winner is None:
                unparsed += 1
                continue
            first += winner == 1
            a, b = q.correct[rep[match.first]], q.correct[rep[match.second]]
            if a != b:
                decisive += 1
                right += (winner == 1) == a
    parsed = judged - unparsed
    return {
        "pairs_judged": judged,
        "unparseable": unparsed,
        "first_position_rate": first / parsed if parsed else None,
        "decisive_pairs": decisive,
        "decisive_accuracy": right / decisive if decisive else None,
        "questions_with_tournament": sum(1 for q in questions if q.selection and q.selection.matches),
    }


def stage_costs(run_dir: Path, ids: set[str], pricing: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    n = max(len(ids), 1)
    for stage in STAGES:
        rows = read_jsonl(run_dir / STAGE_FILES[stage])
        usage = sum((Usage.from_json(r.get("usage")) for i, r in rows.items() if i in ids), Usage())
        record = {
            "latency_s": usage.latency_s,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
        }
        usd = query_cost_usd(record, pricing)
        out[stage] = {
            "calls_per_question": usage.calls / n,
            "input_tokens_per_question": usage.input_tokens / n,
            "output_tokens_per_question": usage.output_tokens / n,
            "gpu_s_per_question": usage.latency_s / n,
            "usd_per_1k_questions": 1000 * usd / n if usd is not None else None,
        }
    return out


def _final_right(q: Question) -> bool:
    return q.right(q.selection.chosen if q.selection else None)


def official_check(run_dir: Path, questions: Sequence[Question]) -> dict[str, Any] | None:
    """Final-answer EX from ``scored.jsonl`` (the official rule) next to ours."""
    path = run_dir / SCORED_FILE
    if not path.is_file():
        return None
    scored = {r["id"]: r for r in read_jsonl(path).values()}
    ours = {q.example.id: _final_right(q) for q in questions}
    common = [i for i in ours if i in scored and scored[i].get("greedy")]
    if not common:
        return None
    official = [bool(scored[i]["greedy"]["correct"]) for i in common]
    return {
        "n": len(common),
        "official_ex": statistics.fmean(official),
        "disagreements": sum(o != ours[i] for o, i in zip(official, common, strict=True)),
    }


def against_baseline(questions: Sequence[Question], baseline_dir: str | Path) -> dict[str, Any]:
    """The agent's answers vs a scored run on the same questions (paired bootstrap and McNemar)."""
    ours = {q.example.id: _final_right(q) for q in questions}
    scored = load_scored(baseline_dir)
    out: dict[str, Any] = {"run": str(baseline_dir)}
    for mode in ("greedy", "sc"):
        theirs = per_question_correct(scored, mode)
        ids = sorted(set(ours) & set(theirs))
        if not ids:
            continue
        comp = paired_comparison([ours[i] for i in ids], [theirs[i] for i in ids])
        out[mode] = {
            "n": comp.n,
            "agent": comp.acc_a,
            "baseline": comp.acc_b,
            "diff": comp.diff,
            "diff_ci95": list(comp.diff_ci),
            "only_agent": comp.only_a,
            "only_baseline": comp.only_b,
            "p_value": comp.p_value,
        }
    return out


def analyze(
    run_dir: str | Path,
    *,
    compare: Mapping[str, str | Path] | None = None,
    baseline: str | Path | None = None,
    examples: list[Example] | None = None,
) -> dict[str, Any]:
    """Compute every analysis for ``run_dir``; writes analysis.json and analysis.md there."""
    run_dir = Path(run_dir)
    config = load_agent_config(run_dir)
    questions = load_questions(run_dir, examples)
    if not questions:
        raise ValueError(f"{run_dir}: no question has finished the select stage")
    result: dict[str, Any] = {
        "run": config.run_name,
        "model": config.name,
        "n": len(questions),
        "gold_failures": sum(1 for q in questions if not q.gold_ok),
        "ablations": ablations(questions),
        "generators": generators(questions),
        "overlap": overlap(questions),
        "candidate_scaling": candidate_scaling(questions),
        "selector": selector_stats(questions),
        "cost": stage_costs(run_dir, {q.example.id for q in questions}, config.pricing),
        "official": official_check(run_dir, questions),
        "compare": {},
        "baseline": against_baseline(questions, baseline) if baseline else None,
    }
    for name, other in (compare or {}).items():
        result["compare"][name] = _by_difficulty(load_questions(other, examples), _final_right)
    (run_dir / ANALYSIS_FILE).write_text(json.dumps(result, indent=2), encoding="utf-8")
    (run_dir / ANALYSIS_MD).write_text(render_markdown(result), encoding="utf-8")
    return result


# ------------------------------------------------------------------ markdown


def _pct(value: float | None) -> str:
    return "–" if value is None else f"{100 * value:.2f}"


def _diff_row(name: str, row: dict[str, Any], base: float | None) -> str:
    total = row.get("total")
    delta = (
        "–" if base is None or total is None or name.startswith("Agentar") else f"{100 * (total - base):+.2f}"
    )
    cells = [_pct(row.get(level)) for level in DIFFICULTIES]
    return f"| {name} | " + " | ".join(cells) + f" | {_pct(total)} | {delta} |"


def render_markdown(result: dict[str, Any]) -> str:
    abl = result["ablations"]
    base = abl["Agentar-Scale (full)"]["total"]
    lines = [
        f"# Agent analysis: {result['model']}",
        "",
        f"{result['n']} questions ({result['gold_failures']} with a failing gold query).",
        "",
        "## Ablations (paper Table 2)",
        "",
        "Computed from this run: the tournament is re-scored with only the candidates and "
        "judgments that remain.",
        "",
        "| Method | Simple | Moderate | Challenging | Total | Δ Total |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lines += [_diff_row(name, row, base) for name, row in abl.items()]
    for name, row in result["compare"].items():
        lines.append(_diff_row(f"{name} (separate run)", row, base))
    lines += [
        "",
        "## Generators (paper Fig. 3)",
        "",
        "| Candidates from | Voting | Selection | Upper bound |",
        "|---|---:|---:|---:|",
    ]
    for name, row in result["generators"].items():
        lines.append(
            f"| {name} | {_pct(row['voting']['total'])} | {_pct(row['selection']['total'])} "
            f"| {_pct(row['upper_bound']['total'])} |"
        )
    ov = result["overlap"]
    lines += [
        "",
        f"Questions some candidate answers correctly: both generators {ov['both']}, "
        f"ICL only {ov['icl_only']}, reasoning only {ov['reasoning_only']}, "
        f"neither {ov['neither']} (paper Fig. 4).",
        "",
        "## pass@k over generated candidates (paper Fig. 6)",
        "",
        "| k | Simple | Moderate | Challenging | Total |",
        "|---:|---:|---:|---:|---:|",
    ]
    for k, row in result["candidate_scaling"].items():
        lines.append(
            f"| {k} | "
            + " | ".join(_pct(row.get(level)) for level in DIFFICULTIES)
            + f" | {_pct(row['total'])} |"
        )
    sel = result["selector"]
    lines += [
        "",
        "## Selector",
        "",
        f"{sel['pairs_judged']} pairs judged over {sel['questions_with_tournament']} questions; "
        f"{sel['unparseable']} judgments had no readable answer.",
        f"Accuracy on the {sel['decisive_pairs']} pairs where exactly one query is right: "
        f"{_pct(sel['decisive_accuracy'])}%. Picked the first position "
        f"{_pct(sel['first_position_rate'])}% of the time.",
        "",
        "## Cost per stage",
        "",
        "| Stage | Calls / question | Input tok | Output tok | GPU s / question | $ / 1k questions |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for stage, c in result["cost"].items():
        usd = c["usd_per_1k_questions"]
        lines.append(
            f"| {stage} | {c['calls_per_question']:.1f} | {c['input_tokens_per_question']:.0f} "
            f"| {c['output_tokens_per_question']:.0f} | {c['gpu_s_per_question']:.2f} "
            f"| {'–' if usd is None else f'${usd:.3f}'} |"
        )
    if result.get("baseline"):
        b = result["baseline"]
        lines += [
            "",
            f"## Against {b['run']} (paired, same questions)",
            "",
            "| Baseline decoding | n | Agent EX | Baseline EX | Δ (pts, 95% CI) | Only agent right "
            "| Only baseline right | McNemar p |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for mode, label in (("greedy", "greedy"), ("sc", "self-consistency")):
            if mode in b:
                c = b[mode]
                lo, hi = c["diff_ci95"]
                p = "<0.001" if c["p_value"] < 0.001 else f"{c['p_value']:.3f}"
                lines.append(
                    f"| {label} | {c['n']} | {_pct(c['agent'])} | {_pct(c['baseline'])} "
                    f"| {100 * c['diff']:+.1f} ({100 * lo:+.1f} to {100 * hi:+.1f}) | {c['only_agent']} "
                    f"| {c['only_baseline']} | {p} |"
                )
    if result.get("official"):
        off = result["official"]
        lines += [
            "",
            f"Official scorer (scored.jsonl): EX {_pct(off['official_ex'])}% on {off['n']} questions; "
            f"{off['disagreements']} disagree with the fingerprint check above.",
        ]
    return "\n".join(lines) + "\n"
