"""Execution-based rewards for GRPO (Arctic-Text2SQL-R1 style).

The reward is deliberately simple. A completion scores 1.0 when its query returns
the same result set as the gold query, 0.1 when it runs but returns something
else, and 0.0 when it has no parsable query or fails to execute. The small
"executable" bonus gives the policy a gradient towards valid SQL early in training
without rewarding it for anything a grader would accept as correct.

Rewards are plain callables with TRL's ``reward_func(prompts, completions,
**columns)`` signature, so they need no TRL import and are unit-tested directly.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from datalens.sql.compare import exec_match
from datalens.sql.executor import ExecResult, execute
from datalens.sql.parse import extract_sql


def completion_text(completion: Any) -> str:
    """Text of a TRL completion, which is a string or a list of chat messages."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, Sequence) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
    return str(completion)


@dataclass(frozen=True)
class RewardConfig:
    correct: float = 1.0
    executable: float = 0.1
    invalid: float = 0.0
    timeout_s: float = 10.0
    num_workers: int = 16
    max_rows: int = 100_000  # rows fetched per rollout (at least twice the gold result's size)


class ExecutionReward:
    """Callable reward that executes each completion's SQL against its database.

    Gold results are cached per ``(db_path, gold_sql)``: GRPO samples many
    completions per prompt and revisits prompts across epochs, so each gold query
    runs once. Completions run in a thread pool; ``sqlite3`` releases the GIL
    while a statement executes.
    """

    __name__ = "execution_reward"  # TRL logs rewards under the function name

    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config = config or RewardConfig()
        self._gold: dict[tuple[str, str], ExecResult] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=self.config.num_workers)

    # The thread pool and lock can't be pickled; recreate them on unpickling.
    def __getstate__(self) -> dict[str, Any]:
        return {"config": self.config, "_gold": self._gold}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__init__(state["config"])  # type: ignore[misc]
        self._gold = state["_gold"]

    def gold_result(self, db_path: str, gold_sql: str) -> ExecResult:
        key = (str(db_path), gold_sql)
        with self._lock:
            cached = self._gold.get(key)
        if cached is not None:
            return cached
        # Gold queries get a generous timeout: a slow reference is not the policy's fault.
        result = execute(db_path, gold_sql, timeout_s=max(self.config.timeout_s, 60.0))
        with self._lock:
            self._gold[key] = result
        return result

    def score(self, text: str, db_path: str, gold_sql: str) -> float:
        cfg = self.config
        sql = extract_sql(text)
        if sql is None:
            return cfg.invalid
        gold = self.gold_result(db_path, gold_sql)
        gold_rows = gold.rows if gold.ok else None
        # Early in training the policy writes the odd accidental cross join; capping
        # the rows fetched keeps 16 of those from exhausting memory mid-run. A query
        # that hits the cap returns far more rows than the answer and is not correct.
        cap = max(cfg.max_rows, 2 * len(gold_rows or ()))
        pred = execute(db_path, sql, timeout_s=cfg.timeout_s, max_rows=cap)
        if not pred.ok or pred.rows is None:
            return cfg.invalid
        if gold_rows is not None and not pred.truncated and exec_match(pred.rows, gold_rows):
            return cfg.correct
        return cfg.executable

    def __call__(
        self,
        prompts: list[Any] | None = None,
        completions: list[Any] | None = None,
        *,
        db_path: list[str],
        gold_sql: list[str],
        log_metric: Callable[[str, float], None] | None = None,
        **_: Any,
    ) -> list[float]:
        texts = [completion_text(c) for c in completions or []]
        futures = [
            self._pool.submit(self.score, text, db, gold)
            for text, db, gold in zip(texts, db_path, gold_sql, strict=True)
        ]
        rewards = [f.result() for f in futures]
        if log_metric is not None and rewards:
            # Shown next to loss/KL in the training logs: the share of rollouts that
            # are correct, and that at least execute.
            n = len(rewards)
            log_metric("sql/exec_accuracy", sum(r == self.config.correct for r in rewards) / n)
            log_metric("sql/executable", sum(r != self.config.invalid for r in rewards) / n)
        return rewards


_FORMAT_RE = re.compile(r"^\s*<think>.+?</think>\s*```sql\s*\n.+?```\s*$", re.DOTALL)


def format_reward(
    prompts: list[Any] | None = None, completions: list[Any] | None = None, **_: Any
) -> list[float]:
    """1.0 when a completion is exactly ``<think>...</think>`` then one sql block.

    Off by default (Arctic-R1 found the execution reward sufficient); enable it
    with a non-zero weight in the GRPO config to tighten output formatting.
    """
    return [1.0 if _FORMAT_RE.match(completion_text(c)) else 0.0 for c in completions or []]


format_reward.__name__ = "format_reward"
