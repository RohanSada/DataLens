"""Phase 1 of an evaluation run: generate completions (needs the model).

Generation and scoring are separate so the GPU-bound part can run on Colab and
the CPU-bound scoring can run anywhere, and so scoring changes never require
regenerating. Output is append-only JSONL, so an interrupted run resumes where
it stopped.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from datalens.data.benchmarks import Example, load_examples, resolve_spec
from datalens.eval.config import EvalConfig
from datalens.inference.backends import Backend, Generation, SamplingConfig, create_backend
from datalens.prompts import prompt_for_example

log = logging.getLogger(__name__)

GENERATIONS_FILE = "generations.jsonl"
CONFIG_FILE = "config.yaml"


def run_dir_for(config: EvalConfig, runs_root: str | Path) -> Path:
    return Path(runs_root) / config.run_name


def load_benchmark(config: EvalConfig) -> list[Example]:
    spec = resolve_spec(config.benchmark, config.data_root)
    return load_examples(spec, limit=config.limit)


def _done_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    done = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                done.add(json.loads(line)["id"])
    return done


def _batches(items: Sequence[Example], size: int) -> Iterator[Sequence[Example]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _record(gen: Generation) -> dict[str, Any]:
    return {
        "completions": [asdict(c) for c in gen.completions],
        "input_tokens": gen.input_tokens,
        "cached_input_tokens": gen.cached_input_tokens,
        "cache_write_tokens": gen.cache_write_tokens,
        "latency_s": round(gen.latency_s, 4),
    }


def generate(
    config: EvalConfig,
    runs_root: str | Path = "runs",
    *,
    backend: Backend | None = None,
    examples: list[Example] | None = None,
) -> Path:
    """Generate greedy and/or sampled completions for every example.

    Returns the run directory. ``backend`` and ``examples`` can be injected (tests
    use a fake backend); otherwise they come from the config.
    """
    run_dir = run_dir_for(config, runs_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / CONFIG_FILE).write_text(config.to_yaml(), encoding="utf-8")

    examples = examples if examples is not None else load_benchmark(config)
    out_path = run_dir / GENERATIONS_FILE
    done = _done_ids(out_path)
    # Group questions by database: schema introspection is cached per database and
    # API prompt caches only live for minutes.
    todo = sorted((e for e in examples if e.id not in done), key=lambda e: e.db_id)
    if not todo:
        log.info("all %d examples already generated in %s", len(examples), out_path)
        return run_dir

    backend = backend or create_backend(
        config.model.backend, config.model.model, **config.model.backend_kwargs
    )
    dec = config.decoding
    greedy = SamplingConfig(n=1, temperature=dec.greedy_temperature, max_tokens=dec.max_tokens, seed=dec.seed)
    sampled = SamplingConfig(
        n=dec.samples, temperature=dec.temperature, top_p=dec.top_p, max_tokens=dec.max_tokens, seed=dec.seed
    )
    prompt_cfg = config.prompt_config()

    log.info("generating %d/%d examples with %s", len(todo), len(examples), config.model.name)
    with out_path.open("a", encoding="utf-8") as fh:
        for batch in _batches(todo, config.batch_size):
            prompts = [prompt_for_example(e, prompt_cfg) for e in batch]
            greedy_out = backend.generate(prompts, greedy) if dec.greedy else [None] * len(batch)
            sampled_out = backend.generate(prompts, sampled) if dec.samples > 0 else [None] * len(batch)
            for example, g, s in zip(batch, greedy_out, sampled_out, strict=True):
                row = {
                    "id": example.id,
                    "db_id": example.db_id,
                    "difficulty": example.difficulty,
                    "greedy": _record(g) if g is not None else None,
                    "samples": _record(s) if s is not None else None,
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            log.info("generated %d examples", len(batch))
    return run_dir
