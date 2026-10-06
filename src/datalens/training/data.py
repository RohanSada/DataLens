"""Building SFT and GRPO training sets from BIRD train.

Filtering matters as much as the algorithm here:

* Gold queries that error or time out would teach the model wrong answers or
  make every rollout score zero.
* Gold queries that return no rows reward any query that returns nothing, which
  GRPO learns to exploit quickly.
* Prompts longer than the token budget would get their schema truncated.
* Optionally, questions the base model already solves on every rollout (or
  never solves) give GRPO zero advantage and only cost compute; pass their ids
  in ``exclude_ids_file``.

``train_fraction`` takes a fixed-seed prefix of a shuffled list, so the 10%
subset is contained in the 25% subset and so on, which keeps learning-curve runs
comparable.
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from datalens.data.benchmarks import Example, load_examples, resolve_spec
from datalens.prompts import PromptConfig, format_answer, prompt_for_example
from datalens.sql.executor import execute
from datalens.training.config import DataSpec, PromptSpec

log = logging.getLogger(__name__)

Messages = list[dict[str, str]]


def subset(examples: list[Example], fraction: float, seed: int) -> list[Example]:
    """Nested fixed-seed subset: smaller fractions are prefixes of larger ones."""
    ordered = sorted(examples, key=lambda e: e.id)
    random.Random(seed).shuffle(ordered)
    keep = max(1, round(len(ordered) * fraction))
    return ordered[:keep]


def gold_status(
    examples: list[Example], *, timeout_s: float, cache_file: Path | None = None, workers: int = 8
) -> dict[str, dict[str, Any]]:
    """Run every gold query once; cache ``{id: {"ok": bool, "rows": int}}`` on disk."""
    cached: dict[str, dict[str, Any]] = {}
    if cache_file is not None and cache_file.is_file():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
    todo = [e for e in examples if e.id not in cached]
    if todo:
        log.info("executing %d gold queries to filter the training set", len(todo))

        def check(example: Example) -> tuple[str, dict[str, Any]]:
            result = execute(example.db_path, example.gold_sql, timeout_s=timeout_s)
            return example.id, {"ok": result.ok, "rows": len(result.rows or [])}

        with ThreadPoolExecutor(max_workers=workers) as pool:
            cached.update(dict(pool.map(check, todo)))
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(cached), encoding="utf-8")
    return cached


def select_examples(spec: DataSpec) -> list[Example]:
    """Load the benchmark split and apply every data filter except prompt length."""
    examples = load_examples(resolve_spec(spec.benchmark, spec.data_root), limit=spec.limit)
    total = len(examples)
    examples = subset(examples, spec.train_fraction, spec.seed)

    if spec.drop_failed_gold or spec.drop_empty_gold:
        cache = Path(".cache/gold") / f"{spec.benchmark}.json"
        status = gold_status(examples, timeout_s=spec.gold_timeout_s, cache_file=cache)
        kept = []
        for e in examples:
            s = status[e.id]
            if spec.drop_failed_gold and not s["ok"]:
                continue
            if spec.drop_empty_gold and s["ok"] and s["rows"] == 0:
                continue
            kept.append(e)
        examples = kept

    if spec.exclude_ids_file:
        excluded = {
            line.strip()
            for line in Path(spec.exclude_ids_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        examples = [e for e in examples if e.id not in excluded]

    log.info("training examples: %d of %d after sampling and filtering", len(examples), total)
    return examples


def build_records(
    examples: list[Example],
    *,
    method: str,
    prompt: PromptSpec,
    schema_cache_dir: str | None,
    count_tokens: Callable[[Messages], int] | None = None,
    max_prompt_tokens: int | None = None,
) -> list[dict[str, Any]]:
    """Turn examples into TRL dataset rows.

    SFT rows are conversational prompt/completion pairs, so loss is computed on
    the answer only. GRPO rows carry ``db_path`` and ``gold_sql`` columns, which
    TRL forwards to the reward function.
    """
    cfg = PromptConfig(
        reasoning=prompt.reasoning,
        num_examples=prompt.num_examples,
        descriptions=prompt.descriptions,
        schema_cache_dir=schema_cache_dir,
    )
    records = []
    too_long = 0
    for example in examples:
        messages = prompt_for_example(example, cfg).messages()
        if (
            count_tokens is not None
            and max_prompt_tokens is not None
            and count_tokens(messages) > max_prompt_tokens
        ):
            too_long += 1
            continue
        row: dict[str, Any] = {"id": example.id, "prompt": messages}
        if method == "sft":
            row["completion"] = [{"role": "assistant", "content": format_answer(example.gold_sql)}]
        else:
            row["db_path"] = str(example.db_path)
            row["gold_sql"] = example.gold_sql
        records.append(row)
    if too_long:
        log.info("dropped %d examples whose prompt exceeds %s tokens", too_long, max_prompt_tokens)
    return records


def exclusion_ids_from_run(scored_path: str | Path, *, low: float = 0.0, high: float = 1.0) -> list[str]:
    """Ids whose sampled pass rate is ``<= low`` or ``>= high`` in a scored run.

    Run the base model on BIRD train with ``samples: 8``, score it, then exclude
    the ids this returns: GRPO learns nothing from questions every rollout gets
    right (or wrong), because all rewards in the group are equal.
    """
    out = []
    for line in Path(scored_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        samples = row.get("samples")
        if not samples or not samples["correct"]:
            continue
        rate = sum(samples["correct"]) / len(samples["correct"])
        if rate <= low or rate >= high:
            out.append(row["id"])
    return out
