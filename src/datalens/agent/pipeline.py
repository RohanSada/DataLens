"""Running the agent over a benchmark: resumable stages, then scorer-ready output.

Each stage appends one JSON line per question to its own file in the run folder,
so a run interrupted by a Colab disconnect continues where it stopped, and each
stage can run in its own process (``datalens agent run --stage reason``), which
guarantees a model's GPU memory is freed before the next one loads.

``finalize`` writes ``generations.jsonl`` and ``config.yaml`` in the format of an
evaluation run: the selected query is the run's answer (scored like a greedy
answer), and the generated candidates are its samples (scored like
self-consistency samples). ``datalens eval score`` and ``datalens eval report``
then treat the agent like any other model.
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from datalens.agent.config import AgentConfig
from datalens.agent.models import ModelPool
from datalens.agent.selection import Candidate, Selection
from datalens.agent.stages import STAGE_FUNCTIONS, AgentContext, Item, Usage
from datalens.agent.understanding import Understanding
from datalens.data.benchmarks import Example, load_examples, resolve_spec
from datalens.eval.generate import CONFIG_FILE, GENERATIONS_FILE

log = logging.getLogger(__name__)

STAGES = ("understand", "reason", "icl", "refine", "select")
STAGE_FILES = {
    "understand": "understanding.jsonl",
    "reason": "reasoning.jsonl",
    "icl": "icl.jsonl",
    "refine": "refinement.jsonl",
    "select": "selection.jsonl",
}
AGENT_CONFIG_FILE = "agent_config.yaml"


def stage_enabled(config: AgentConfig, stage: str) -> bool:
    if stage == "understand":
        return config.understanding.enabled
    if stage == "reason":
        return config.reasoning.enabled
    if stage == "icl":
        return config.icl.enabled
    if stage == "refine":
        return config.refinement.fixer or config.refinement.revisor
    return True


def stage_models(config: AgentConfig, stage: str) -> set[str]:
    """The models a stage will call (others can be freed before it runs)."""
    roles = config.roles
    if stage == "understand":
        u = config.understanding
        needs = u.llm or (u.example_skeletons == "llm" and u.examples > 0 and config.icl.enabled)
        return {roles.understanding.model} if needs else set()
    if stage == "reason":
        return {roles.reasoning_generator.model}
    if stage == "icl":
        return {roles.icl_generator.model}
    if stage == "refine":
        r = config.refinement
        return ({roles.fixer.model} if r.fixer else set()) | ({roles.revisor.model} if r.revisor else set())
    return {roles.selector.model} if config.selection.method == "tournament" else set()


def read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    """Rows by id; a later row for the same id replaces an earlier one."""
    rows: dict[str, dict[str, Any]] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["id"]] = row
    return rows


def _batches(items: Sequence[Example], size: int) -> Iterator[Sequence[Example]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def select_examples(config: AgentConfig) -> list[Example]:
    examples = load_examples(resolve_spec(config.benchmark, config.data_root), limit=config.limit)
    if config.sample and config.sample < len(examples):
        keep = set(random.Random(config.seed).sample(range(len(examples)), config.sample))
        examples = [e for i, e in enumerate(examples) if i in keep]
    return examples


class AgentRun:
    """One agent run: a config, its run folder and the models it loads."""

    def __init__(
        self,
        config: AgentConfig,
        runs_root: str | Path = "runs",
        *,
        pool: ModelPool | None = None,
        examples: list[Example] | None = None,
    ) -> None:
        self.config = config
        self.dir = Path(runs_root) / config.run_name
        self.pool = pool or ModelPool(config.models, max_model_len=config.max_model_len)
        self.ctx = AgentContext(config, self.pool)
        self._examples = examples

    def examples(self) -> list[Example]:
        if self._examples is None:
            self._examples = select_examples(self.config)
        return self._examples

    # -------------------------------------------------------------- state

    def load_state(self) -> dict[str, dict[str, dict[str, Any]]]:
        return {stage: read_jsonl(self.dir / name) for stage, name in STAGE_FILES.items()}

    @staticmethod
    def build_item(example: Example, state: dict[str, dict[str, dict[str, Any]]]) -> Item:
        item = Item(example=example)
        if row := state["understand"].get(example.id):
            item.understanding = Understanding.from_json(row["understanding"])
        for stage, attr in (("reason", "reasoning"), ("icl", "icl"), ("refine", "refined")):
            if row := state[stage].get(example.id):
                setattr(item, attr, [Candidate.from_json(c) for c in row["candidates"]])
        if row := state["select"].get(example.id):
            item.selection = Selection.from_json(row["selection"])
        return item

    @staticmethod
    def _record(stage: str, item: Item) -> dict[str, Any]:
        row: dict[str, Any] = {"id": item.id, "usage": item.usage.to_json()}
        if item.notes:
            row["notes"] = item.notes
        if stage == "understand":
            row["understanding"] = item.understanding.to_json()
        elif stage == "select":
            assert item.selection is not None
            row["selection"] = item.selection.to_json()
        else:
            attr = {"reason": "reasoning", "icl": "icl", "refine": "refined"}[stage]
            row["candidates"] = [c.to_json() for c in getattr(item, attr)]
        return row

    # -------------------------------------------------------------- running

    def run(self, stages: Sequence[str] | None = None) -> Path:
        """Run the given stages (all by default) for every question not done yet, then finalize."""
        stages = list(stages or STAGES)
        unknown = [s for s in stages if s not in STAGES]
        if unknown:
            raise ValueError(f"unknown stage(s) {unknown}; expected some of {list(STAGES)}")
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / AGENT_CONFIG_FILE).write_text(self.config.to_yaml(), encoding="utf-8")
        for stage in STAGES:
            if stage in stages:
                self.run_stage(stage)
        self.finalize()
        return self.dir

    def run_stage(self, stage: str) -> None:
        if not stage_enabled(self.config, stage):
            log.info("stage %s is disabled in the config; skipping", stage)
            return
        path = self.dir / STAGE_FILES[stage]
        done = set(read_jsonl(path))
        todo = sorted((e for e in self.examples() if e.id not in done), key=lambda e: (e.db_id, e.id))
        if not todo:
            log.info("stage %s: all %d questions done", stage, len(self.examples()))
            return
        if self.config.release_models:
            self.pool.release(keep=stage_models(self.config, stage))
        state = self.load_state()
        fn = STAGE_FUNCTIONS[stage]
        log.info("stage %s: %d questions to do", stage, len(todo))
        finished = 0
        start = time.perf_counter()
        for chunk in _batches(todo, self.config.batch_size):
            items = [self.build_item(e, state) for e in chunk]
            fn(items, self.ctx)
            with path.open("a", encoding="utf-8") as fh:
                for item in items:
                    fh.write(json.dumps(self._record(stage, item), ensure_ascii=False) + "\n")
            finished += len(items)
            log.info(
                "stage %s: %d/%d questions (%.0fs)", stage, finished, len(todo), time.perf_counter() - start
            )

    # -------------------------------------------------------------- output

    def _slots(self) -> tuple[int, int]:
        """Reasoning and ICL candidates per question when every prompt fits."""
        reasoning = self.config.reasoning.n if self.config.reasoning.enabled else 0
        icl = sum(v.n for v in self.config.icl.variants) if self.config.icl.enabled else 0
        return reasoning, icl

    def finalize(self) -> Path:
        """Write ``generations.jsonl`` and ``config.yaml`` for questions that finished selection."""
        state = self.load_state()
        slots_r, slots_i = self._slots()
        rows = []
        for example in self.examples():
            if example.id not in state["select"]:
                continue
            item = self.build_item(example, state)
            usage = {
                stage: Usage.from_json((state[stage].get(example.id) or {}).get("usage")) for stage in STAGES
            }
            pool = item.pool()
            chosen = item.selection.chosen if item.selection else None
            if chosen is None:  # nothing executed: answer with the first query there is
                chosen = next((i for i, c in enumerate(pool) if c.sql), None)
            final_sql = pool[chosen].sql if chosen is not None else None
            total = sum(usage.values(), Usage())
            generation = usage["understand"] + usage["reason"] + usage["icl"]

            # Samples are the generated candidates in fixed slots (padded when a prompt was
            # skipped), so maj@k and pass@k compare like with like across questions.
            def padded(cands: list[Candidate], size: int) -> list[dict[str, Any]]:
                out = [_completion(c.sql, c.output_tokens, c.finish_reason) for c in cands[:size]]
                return out + [_completion(None, 0, "skipped")] * (size - len(out))

            samples = padded(item.reasoning, slots_r) + padded(item.icl, slots_i)
            rows.append(
                {
                    "id": example.id,
                    "db_id": example.db_id,
                    "difficulty": example.difficulty,
                    "greedy": {
                        "completions": [_completion(final_sql, total.output_tokens, "stop")],
                        "input_tokens": total.input_tokens,
                        "cached_input_tokens": 0,
                        "cache_write_tokens": 0,
                        "latency_s": round(total.latency_s, 4),
                    },
                    "samples": {
                        "completions": samples,
                        "input_tokens": generation.input_tokens,
                        "cached_input_tokens": 0,
                        "cache_write_tokens": 0,
                        "latency_s": round(generation.latency_s, 4),
                    }
                    if samples
                    else None,
                }
            )
        out = self.dir / GENERATIONS_FILE
        self.dir.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        (self.dir / CONFIG_FILE).write_text(
            self.config.eval_config(slots_r + slots_i).to_yaml(), encoding="utf-8"
        )
        log.info("wrote %d answers to %s", len(rows), out)
        return out


def _completion(sql: str | None, output_tokens: int | None, finish_reason: str | None) -> dict[str, Any]:
    return {
        "text": f"```sql\n{sql}\n```" if sql else "",
        "output_tokens": output_tokens or 0,
        "finish_reason": finish_reason,
    }


def load_agent_config(run_dir: str | Path) -> AgentConfig:
    return AgentConfig.from_yaml(Path(run_dir) / AGENT_CONFIG_FILE)
