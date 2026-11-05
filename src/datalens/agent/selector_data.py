"""Training data for the GRPO selector (paper section 3.4.2).

Run the agent's generation stages on BIRD train, then turn each question whose
pool holds both a correct and a wrong result into selection prompts: the
correct query against the most-voted wrong one (a hard negative) and against
another wrong one, in random positions. The prompt is exactly what the selector
sees at inference, and the reward is 1 for picking the correct position. The
paper builds 8.5k such samples from BIRD train.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

from datalens.agent.execution import QueryRunner
from datalens.agent.models import ModelPool
from datalens.agent.pipeline import AgentRun, load_agent_config
from datalens.agent.prompts import ShownCandidate, selector_prompt
from datalens.agent.selection import Candidate, ResultGroup, group_by_result
from datalens.agent.stages import AgentContext
from datalens.agent.values import render_matches
from datalens.data.benchmarks import Example

log = logging.getLogger(__name__)


def build_selector_records(
    run_dir: str | Path,
    *,
    examples: list[Example] | None = None,
    pairs_per_question: int = 2,
    max_records: int | None = 8500,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Selection prompts with the position of the correct query as ``label`` ("1" or "2")."""
    run_dir = Path(run_dir)
    config = load_agent_config(run_dir)
    run = AgentRun(config, run_dir.parent, examples=examples)
    state = run.load_state()
    done = set(state["refine"]) | set(state["icl"]) | set(state["reason"])
    finished = [e for e in run.examples() if e.id in done]
    ctx = AgentContext(config, ModelPool({}))  # renders schemas only; calls no model
    runner = QueryRunner(timeout_s=120.0, workers=config.execution.workers)
    gold = runner.run_many([(e.db_path, e.gold_sql) for e in finished])

    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    usable = 0
    for example, gold_outcome in zip(finished, gold, strict=True):
        if not gold_outcome.ok:
            continue
        item = run.build_item(example, state)
        pool = item.pool()
        groups = group_by_result(pool)
        if config.selection.max_candidates:
            groups = groups[: config.selection.max_candidates]  # the groups that would enter the tournament
        right = [g for g in groups if g.key == gold_outcome.key]
        wrong = [g for g in groups if g.key != gold_outcome.key]
        if not right or not wrong:
            continue
        usable += 1
        negatives = wrong[:1] + rng.sample(wrong[1:], min(len(wrong) - 1, max(pairs_per_question - 1, 0)))
        light = ctx.light_schema(item)
        matched = render_matches(item.understanding.matches)

        for negative in negatives[:pairs_per_question]:
            positive = right[0]
            first, second = (positive, negative) if rng.random() < 0.5 else (negative, positive)
            prompt = selector_prompt(
                light,
                matched,
                example.question,
                example.evidence,
                [_shown(pool, first), _shown(pool, second)],
            )
            records.append(
                {
                    "id": f"{example.id}:{first.key[:8]}:{second.key[:8]}",
                    "db_id": example.db_id,
                    "prompt": prompt.messages(),
                    "label": "1" if first is positive else "2",
                }
            )
    rng.shuffle(records)
    if max_records is not None:
        records = records[:max_records]
    log.info(
        "%d selector samples from %d of %d questions with both right and wrong results",
        len(records),
        usable,
        len(finished),
    )
    return records


def _shown(pool: list[Candidate], group: ResultGroup) -> ShownCandidate:
    cand = pool[group.representative]
    return ShownCandidate(sql=cand.sql or "", result=cand.outcome.describe())


def write_records(records: list[dict[str, Any]], out: str | Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return out
