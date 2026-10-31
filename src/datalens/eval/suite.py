"""Running a list of evaluation runs with as few model loads as possible.

A suite file lists runs (a config plus overrides). Consecutive runs that use the
same vLLM engine share one loaded model: the plain harness and the agent both
wrap the same ``vllm.LLM``. Each engine runs in its own subprocess, so its GPU
memory is fully released before the next model loads, and a model that fails to
load doesn't stop the rest of the suite. Scoring, review files and the report
run afterwards on CPU.

```yaml
runs:
  - config: configs/eval/frontier-qwen3-coder-30b-a3b.yaml
    run_name: tune-base-qwen3-coder-30b
    set: [benchmark=bird-train, sample=300, drop_bad_gold=true]
reports:
  reports/tune-round1: [tune-base-qwen3-coder-30b]   # report folder: run names (empty = all)
```
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from datalens.config_utils import load_yaml
from datalens.eval.config import EvalConfig
from datalens.eval.generate import GENERATIONS_FILE, _done_ids, generate, load_benchmark, run_dir_for

log = logging.getLogger(__name__)


class SuiteRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config: str
    run_name: str | None = None
    set: list[str] = Field(default_factory=list)


class Suite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    runs: list[SuiteRun]
    reports: dict[str, list[str]] = Field(
        default_factory=dict, description="Report folder -> run names to compare there (empty list = all)."
    )
    review: bool = Field(True, description="Write review.jsonl (failures with result previews) per run.")


def load_suite(path: str | Path) -> tuple[Suite, list[EvalConfig]]:
    suite = Suite.model_validate(load_yaml(path))
    configs = [EvalConfig.from_yaml(r.config, r.set, run_name=r.run_name) for r in suite.runs]
    names = [c.run_name for c in configs]
    if len(set(names)) != len(names):
        raise ValueError(f"run names must be unique in a suite: {names}")
    return suite, configs


_CHAT_ONLY_KWARGS = ("chat_template_kwargs",)


def engine_kwargs(config: EvalConfig) -> dict[str, Any]:
    kwargs = dict(config.model.backend_kwargs)
    if config.agent is not None:
        kwargs.update(config.agent.chat_kwargs)
    for key in _CHAT_ONLY_KWARGS:
        kwargs.pop(key, None)
    return kwargs


def engine_key(config: EvalConfig) -> str | None:
    """Runs with the same key can share one loaded vLLM engine; ``None`` for API backends."""
    if config.model.backend != "vllm":
        return None
    return json.dumps([config.model.model, engine_kwargs(config)], sort_keys=True)


def is_complete(config: EvalConfig, runs_root: str | Path) -> bool:
    path = run_dir_for(config, runs_root) / GENERATIONS_FILE
    done = _done_ids(path)
    return bool(done) and {e.id for e in load_benchmark(config)} <= done


def generate_group(configs: list[EvalConfig], runs_root: str | Path) -> None:
    """Generate every run in ``configs`` (which share an engine) with one model load."""
    todo = [c for c in configs if not is_complete(c, runs_root)]
    if not todo:
        return
    first = todo[0]
    if first.model.backend != "vllm":
        for config in todo:
            generate(config, runs_root)
        return

    from vllm import LLM

    from datalens.agent.chat import VLLMChat
    from datalens.agent.pipeline import generate_agentic
    from datalens.inference.backends import VLLMBackend

    kwargs = engine_kwargs(first)
    kwargs.setdefault("enable_prefix_caching", True)
    llm = LLM(model=first.model.model, **kwargs)
    for config in todo:
        log.info("suite: generating %s", config.run_name)
        if config.agent is not None:
            template_kwargs = config.agent.chat_kwargs.get(
                "chat_template_kwargs"
            ) or config.model.backend_kwargs.get("chat_template_kwargs")
            chat = VLLMChat.from_llm(llm, config.model.model, chat_template_kwargs=template_kwargs)
            generate_agentic(config, runs_root, model=chat)
        else:
            generate(config, runs_root, backend=VLLMBackend.from_llm(llm, config.model.model))


def groups_of(configs: list[EvalConfig]) -> list[list[int]]:
    """Indices of consecutive runs that share an engine."""
    groups: list[list[int]] = []
    last: str | None = None
    for i, config in enumerate(configs):
        key = engine_key(config)
        if groups and key is not None and key == last:
            groups[-1].append(i)
        else:
            groups.append([i])
        last = key
    return groups


def run_suite(path: str | Path, runs_root: str | Path = "runs", *, isolate: bool = True) -> list[Path]:
    """Generate (one subprocess per engine), then score, review and report. Returns run folders."""
    from datalens.eval.metrics import write_metrics
    from datalens.eval.review import write_review
    from datalens.eval.score import score_run

    suite, configs = load_suite(path)
    failed: list[str] = []
    for group in groups_of(configs):
        names = [configs[i].run_name for i in group]
        if all(is_complete(configs[i], runs_root) for i in group):
            log.info("suite: %s already generated", ", ".join(names))
            continue
        if isolate:
            cmd = [
                sys.executable,
                "-m",
                "datalens.cli",
                "eval",
                "suite-generate",
                str(path),
                "--runs",
                str(runs_root),
                *[arg for i in group for arg in ("--index", str(i))],
            ]
            log.info("suite: generating %s in a subprocess", ", ".join(names))
            if subprocess.run(cmd, check=False).returncode != 0:
                log.error("suite: generation failed for %s; continuing with the rest", ", ".join(names))
                failed.extend(names)
        else:
            generate_group([configs[i] for i in group], runs_root)

    run_dirs: list[Path] = []
    scored_names: list[str] = []
    for config in configs:
        run_dir = run_dir_for(config, runs_root)
        if not (run_dir / GENERATIONS_FILE).is_file():
            continue
        score_run(run_dir)
        summary = write_metrics(run_dir)
        greedy = summary.get("greedy") or {}
        log.info("suite: %s EX %.1f%%", config.run_name, 100 * greedy.get("ex", 0.0))
        if suite.review:
            write_review(run_dir)
        run_dirs.append(run_dir)
        scored_names.append(config.run_name)

    from datalens.eval.report import build_report

    for out_dir, names in suite.reports.items():
        chosen = [d for d, n in zip(run_dirs, scored_names, strict=True) if not names or n in names]
        if chosen:
            build_report(chosen, out_dir, plots=True)
    if failed:
        log.error("suite: these runs did not finish generating: %s", ", ".join(failed))
    return run_dirs
