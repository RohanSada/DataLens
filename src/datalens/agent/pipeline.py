"""Running the agent over a benchmark split.

For each question: render the schema with BIRD's column descriptions, look up
the question's literal values in the database, run several agent attempts that
can query the database, then pick one answer by result-based voting and, when
the attempts disagree, a model-judged comparison.

Questions are processed in chunks sorted by database. Every attempt of every
question in a chunk takes its next turn in one batched model call, and the
selector judgements of a chunk are one more call.

Output goes to the run folder in the same ``generations.jsonl`` format as the
plain harness, so ``datalens eval score`` grades it with the unchanged scorer:

* ``greedy`` holds the chosen answer (graded as the headline EX),
* ``samples`` holds each attempt's answer (giving maj@k and pass@k),
* ``agent`` holds the decision record, and ``traces.jsonl`` every conversation.
"""

from __future__ import annotations

import json
import logging
import time
import zlib
from collections.abc import Iterator, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from datalens.agent.chat import ChatModel, ChatSampling, Conversation, create_chat_model
from datalens.agent.config import AgentSpec
from datalens.agent.prompts import AGENT_SYSTEM, DIRECT_SYSTEM, question_block, schema_block
from datalens.agent.protocol import STOP_SEQUENCES
from datalens.agent.rollout import Attempt, Database, Usage, attempt_summary, run_attempts
from datalens.agent.select import (
    Selection,
    finish_selection,
    preliminary,
    selector_requests,
)
from datalens.agent.values import ValueIndex
from datalens.data.benchmarks import Example
from datalens.data.schema import load_schema
from datalens.eval.config import EvalConfig
from datalens.eval.generate import CONFIG_FILE, GENERATIONS_FILE, _done_ids, load_benchmark, run_dir_for

log = logging.getLogger(__name__)

TRACES_FILE = "traces.jsonl"


def render_schema(example: Example, spec: AgentSpec, cache_dir: str | None) -> str:
    """Schema text, degraded step by step until it fits ``max_schema_chars``."""
    schema = load_schema(example.db_path, num_examples=spec.num_examples, cache_dir=cache_dir)
    attempts = [
        {"examples": spec.num_examples > 0, "descriptions": spec.descriptions, "max_description_chars": 200},
        {"examples": spec.num_examples > 0, "descriptions": spec.descriptions, "max_description_chars": 80},
        {"examples": spec.num_examples > 0, "descriptions": False},
        {"examples": False, "descriptions": False},
    ]
    text = ""
    for kwargs in attempts:
        text = schema.render(**kwargs)  # type: ignore[arg-type]
        if len(text) <= spec.max_schema_chars:
            return text
    return text


class _Resources:
    """Schemas, value indexes and database handles, reused across a run's chunks."""

    def __init__(self, spec: AgentSpec, schema_cache: str | None, value_cache: str | None) -> None:
        self.spec = spec
        self.schema_cache = schema_cache
        self.value_cache = value_cache
        self._schemas: dict[Path, str] = {}
        self._values: dict[Path, ValueIndex] = {}
        self._dbs: dict[Path, Database] = {}

    def schema(self, example: Example) -> str:
        if example.db_path not in self._schemas:
            self._schemas[example.db_path] = render_schema(example, self.spec, self.schema_cache)
        return self._schemas[example.db_path]

    def values(self, example: Example) -> ValueIndex | None:
        if not (self.spec.value_hints or self.spec.tools):
            return None
        if example.db_path not in self._values:
            self._values[example.db_path] = ValueIndex.load(example.db_path, self.value_cache)
        return self._values[example.db_path]

    def database(self, example: Example) -> Database:
        if example.db_path not in self._dbs:
            # Fresh handle (and query cache) per database; older ones are dropped to bound memory.
            self._dbs = {
                example.db_path: Database(
                    example.db_path,
                    self.values(example),
                    timeout_s=self.spec.exec_timeout_s,
                    preview_rows=self.spec.preview_rows,
                )
            }
        return self._dbs[example.db_path]


def _seed(spec: AgentSpec, example: Example, index: int) -> int:
    return zlib.crc32(f"{spec.seed}:{example.id}:{index}".encode())


def build_attempts(
    example: Example, spec: AgentSpec, res: _Resources
) -> tuple[list[Attempt], str, list[str]]:
    schema = schema_block(res.schema(example))
    values = res.values(example)
    hints = (
        values.hints(example.question, example.evidence) if (values is not None and spec.value_hints) else []
    )
    question = question_block(example.question, example.evidence, hints)
    system = AGENT_SYSTEM if spec.tools else DIRECT_SYSTEM
    db = res.database(example)
    attempts = []
    for i in range(spec.rollouts):
        greedy = spec.greedy_first and i == 0
        sampling = ChatSampling(
            temperature=0.0 if greedy else spec.temperature,
            top_p=None if greedy else spec.top_p,
            max_tokens=spec.max_tokens_per_turn,
            seed=_seed(spec, example, i),
            stop=STOP_SEQUENCES if spec.stop_sequences else (),
        )
        messages: Conversation = [
            {"role": "system", "content": system},
            {"role": "user", "content": schema + question, "parts": [schema, question]},
        ]
        attempts.append(
            Attempt(
                question_id=example.id,
                index=i,
                messages=messages,
                sampling=sampling,
                db=db,
                max_turns=spec.max_turns if spec.tools else 1,
                tools=spec.tools,
                retry_empty=spec.retry_empty,
                max_context_chars=spec.max_context_chars,
            )
        )
    return attempts, schema, hints


def _usage_record(usage: Usage, latency_s: float) -> dict[str, Any]:
    return {
        "input_tokens": usage.input_tokens,
        "cached_input_tokens": usage.cached_input_tokens,
        "cache_write_tokens": usage.cache_write_tokens,
        "latency_s": round(latency_s, 4),
    }


def _sql_text(sql: str | None) -> str:
    return f"```sql\n{sql}\n```" if sql else ""


def run_chunk(
    model: ChatModel, examples: Sequence[Example], spec: AgentSpec, res: _Resources
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Answer one chunk of questions. Returns generation rows and trace rows."""
    start = time.perf_counter()
    per_question: list[tuple[Example, list[Attempt], str, list[str]]] = []
    for example in examples:
        attempts, schema, hints = build_attempts(example, spec, res)
        per_question.append((example, attempts, schema, hints))

    all_attempts = [a for _, attempts, _, _ in per_question for a in attempts]
    run_attempts(model, all_attempts)

    selections: list[Selection] = []
    pending: list[tuple[int, list[int], Conversation, ChatSampling]] = []
    for qi, (example, attempts, schema, _hints) in enumerate(per_question):
        selection = preliminary(attempts, spec)
        selections.append(selection)
        if selection.method == "selector":
            for order, conv, samp in selector_requests(
                example.id,
                example.question,
                example.evidence,
                schema,
                selection,
                res.database(example),
                attempts,
                spec,
            ):
                pending.append((qi, order, conv, samp))

    selector_usage = [Usage() for _ in per_question]
    selector_texts: list[list[dict[str, Any]]] = [[] for _ in per_question]
    if pending:
        replies = model.chat([p[2] for p in pending], [p[3] for p in pending])
        judgements: dict[int, list[tuple[list[int], str]]] = {}
        for (qi, order, _conv, _samp), reply in zip(pending, replies, strict=True):
            selector_usage[qi].add(reply)
            judgements.setdefault(qi, []).append((order, reply.text))
            selector_texts[qi].append({"order": order, "text": reply.text})
        for qi, items in judgements.items():
            finish_selection(selections[qi], items, len(per_question[qi][1]), spec)

    latency = (time.perf_counter() - start) / max(len(examples), 1)
    rows, traces = [], []
    for qi, (example, attempts, _schema, hints) in enumerate(per_question):
        selection = selections[qi]
        attempt_usage = Usage()
        for a in attempts:
            attempt_usage = attempt_usage.merge(a.usage)
        total = attempt_usage.merge(selector_usage[qi])
        rows.append(
            {
                "id": example.id,
                "db_id": example.db_id,
                "difficulty": example.difficulty,
                "greedy": {
                    "completions": [
                        {
                            "text": _sql_text(selection.sql),
                            "output_tokens": total.output_tokens,
                            "finish_reason": selection.method,
                        }
                    ],
                    **_usage_record(total, latency),
                },
                "samples": {
                    "completions": [
                        {
                            "text": _sql_text(a.answer_sql),
                            "output_tokens": a.usage.output_tokens,
                            "finish_reason": a.status,
                        }
                        for a in attempts
                    ],
                    **_usage_record(attempt_usage, latency),
                },
                "agent": {
                    "method": selection.method,
                    "chosen_group": selection.chosen_group,
                    "groups": [
                        {"members": g.members, "rows": g.rows, "empty": g.empty, "sql": g.sql}
                        for g in selection.groups
                    ],
                    "shown": selection.shown,
                    "scores": selection.scores,
                    "judgements": selection.judgements,
                    "hints": len(hints),
                    "model_calls": total.calls,
                    "attempts": [attempt_summary(a) for a in attempts],
                },
            }
        )
        traces.append(
            {
                "id": example.id,
                "question": example.question,
                "evidence": example.evidence,
                "hints": hints,
                "attempts": [
                    {"index": a.index, "status": a.status, "messages": a.transcript} for a in attempts
                ],
                "selector": selector_texts[qi],
            }
        )
    return rows, traces


def _chunks(items: Sequence[Example], size: int) -> Iterator[Sequence[Example]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def generate_agentic(
    config: EvalConfig,
    runs_root: str | Path = "runs",
    *,
    model: ChatModel | None = None,
    examples: list[Example] | None = None,
    value_cache_dir: str | None = ".cache/values",
) -> Path:
    """Run the agent on every example not yet in the run's ``generations.jsonl``."""
    spec = config.agent
    if spec is None:
        raise ValueError("config has no agent section")
    run_dir = run_dir_for(config, runs_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / CONFIG_FILE).write_text(config.to_yaml(), encoding="utf-8")

    examples = examples if examples is not None else load_benchmark(config)
    out_path = run_dir / GENERATIONS_FILE
    done = _done_ids(out_path)
    todo = sorted((e for e in examples if e.id not in done), key=lambda e: (e.db_id, e.id))
    if not todo:
        log.info("all %d examples already generated in %s", len(examples), out_path)
        return run_dir

    if model is None:
        kwargs = {**config.model.backend_kwargs, **spec.chat_kwargs}
        model = create_chat_model(config.model.backend, config.model.model, **kwargs)
    res = _Resources(spec, config.schema_cache_dir, value_cache_dir)

    log.info(
        "agent: %d/%d examples with %s, %d attempts each",
        len(todo),
        len(examples),
        config.model.name,
        spec.rollouts,
    )
    finished = len(examples) - len(todo)
    with (
        out_path.open("a", encoding="utf-8") as fh,
        (run_dir / TRACES_FILE).open("a", encoding="utf-8") as tf,
    ):
        for chunk in _chunks(todo, spec.chunk_size):
            t0 = time.perf_counter()
            rows, traces = run_chunk(model, chunk, spec, res)
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            for trace in traces:
                tf.write(json.dumps(trace, ensure_ascii=False) + "\n")
            fh.flush()
            tf.flush()
            finished += len(chunk)
            log.info(
                "agent: %d/%d done (%.0fs for this chunk)", finished, len(examples), time.perf_counter() - t0
            )
    return run_dir


__all__ = ["TRACES_FILE", "asdict", "generate_agentic", "render_schema", "run_chunk"]
