"""Agent attempts and the lockstep loop that drives them.

An :class:`Attempt` is one conversation between the model and a database: the
model writes an action, the harness executes it and appends the observation,
until the model answers or runs out of turns. :func:`run_attempts` advances every
unfinished attempt by one turn per batched model call.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from datalens.agent.chat import ChatModel, ChatResult, ChatSampling, Conversation
from datalens.agent.prompts import FORMAT_REMINDER, LAST_TURN, TURNS_LEFT, format_rows
from datalens.agent.protocol import close_turn, parse_action
from datalens.agent.values import ValueIndex, format_matches
from datalens.sql.compare import result_key
from datalens.sql.executor import ErrorKind, ExecResult, execute
from datalens.sql.parse import is_read_only

log = logging.getLogger(__name__)

READ_ONLY_ERROR = "only a single read-only SELECT (or WITH ... SELECT) query is allowed"
MAX_FINAL_ROWS = 20_000  # enough to fingerprint a result for voting without exhausting memory


class Database:
    """Read-only access to one database, shared by every attempt on it.

    Results are cached by query text: attempts on the same question often run the
    same exploratory queries.
    """

    def __init__(
        self,
        db_path: str | Path,
        values: ValueIndex | None = None,
        *,
        timeout_s: float = 10.0,
        preview_rows: int = 10,
    ) -> None:
        self.db_path = Path(db_path)
        self.values = values
        self.timeout_s = timeout_s
        self.preview_rows = preview_rows
        self._cache: dict[tuple[str, int], ExecResult] = {}
        self._lock = threading.Lock()

    def run(self, sql: str, *, max_rows: int = 200, timeout_s: float | None = None) -> ExecResult:
        key = (sql, max_rows)
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            return cached
        if not is_read_only(sql):
            result = ExecResult(ok=False, error=READ_ONLY_ERROR, error_kind=ErrorKind.SYNTAX)
        else:
            result = execute(self.db_path, sql, timeout_s=timeout_s or self.timeout_s, max_rows=max_rows)
        with self._lock:
            self._cache[key] = result
        return result

    def run_final(self, sql: str) -> ExecResult:
        return self.run(sql, max_rows=MAX_FINAL_ROWS, timeout_s=max(self.timeout_s, 30.0))

    def describe(self, result: ExecResult) -> str:
        if not result.ok:
            return f"Error: {result.error}"
        return format_rows(
            result.columns,
            result.rows or [],
            truncated=result.truncated,
            max_rows=self.preview_rows,
        )

    def search(self, text: str) -> str:
        if self.values is None:
            return "Value search is not available for this database."
        return format_matches(self.values.search(text), text)


def fingerprint(result: ExecResult | None) -> str | None:
    """Result key used to group attempts that return the same rows."""
    if result is None or not result.ok or result.rows is None:
        return None
    key = result_key(result.rows)
    return key + ("+" if result.truncated else "")


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0

    def add(self, r: ChatResult) -> None:
        self.calls += 1
        self.input_tokens += r.input_tokens
        self.cached_input_tokens += r.cached_input_tokens
        self.cache_write_tokens += r.cache_write_tokens
        self.output_tokens += r.output_tokens

    def merge(self, other: Usage) -> Usage:
        return Usage(
            self.calls + other.calls,
            self.input_tokens + other.input_tokens,
            self.cached_input_tokens + other.cached_input_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
            self.output_tokens + other.output_tokens,
        )


@dataclass
class Attempt:
    """One agent conversation for one question."""

    question_id: str
    index: int
    messages: Conversation
    sampling: ChatSampling
    db: Database
    max_turns: int = 8
    tools: bool = True
    retry_empty: bool = True
    max_context_chars: int = 90_000

    turns: int = 0
    done: bool = False
    status: str = "running"  # answered | answered_after_limit | fallback | no_answer | failed
    answer_sql: str | None = None
    answer_result: ExecResult | None = None
    actions: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    _prompt_len: int = 0
    _empty_retried: bool = False
    _fallback: tuple[str, ExecResult] | None = None
    _force_answer: bool = False

    def __post_init__(self) -> None:
        self._prompt_len = len(self.messages)

    @property
    def transcript(self) -> list[dict[str, str]]:
        """The turns after the prompt (for traces)."""
        return [{"role": m["role"], "content": m["content"]} for m in self.messages[self._prompt_len :]]

    @property
    def key(self) -> str | None:
        return fingerprint(self.answer_result)

    def _chars(self) -> int:
        return sum(len(m["content"]) for m in self.messages)

    def _finish(self, status: str) -> None:
        self.done = True
        if self.answer_sql is None and self._fallback is not None:
            # Out of turns without an answer: the last query that ran and returned rows
            # is usually the agent's working draft.
            self.answer_sql, self.answer_result = self._fallback
            status = "fallback"
        elif self.answer_sql is None:
            status = "no_answer"
        self.status = status

    def _observe(self, text: str) -> None:
        remaining = self.max_turns - self.turns
        if self._chars() + len(text) > self.max_context_chars:
            self._force_answer = True
        if remaining == 1 or self._force_answer:
            text = f"{text}\n\n{LAST_TURN}"
        elif remaining <= 3:
            text = f"{text}\n\n{TURNS_LEFT.format(n=remaining)}"
        self.messages.append({"role": "user", "content": text})

    def step(self, reply: ChatResult) -> None:
        """Apply one model turn: execute its action and append the observation."""
        self.turns += 1
        self.usage.add(reply)
        self.messages.append({"role": "assistant", "content": close_turn(reply.text, reply.stop_sequence)})
        action = parse_action(reply.text, reply.stop_sequence)
        self.actions.append(action.kind)
        out_of_turns = self.turns >= self.max_turns or self._force_answer

        if action.kind == "answer" and action.sql:
            sql = action.sql
            if not self.tools:
                self.answer_sql, self.answer_result = sql, self.db.run_final(sql)
                return self._finish("answered")
            result = self.db.run_final(sql)
            if result.ok:
                self.answer_sql, self.answer_result = sql, result
                empty = not result.rows
                if empty and self.retry_empty and not self._empty_retried and not out_of_turns:
                    self._empty_retried = True
                    return self._observe(
                        "<result>\nYour answer returned 0 rows.\n</result>\n"
                        "Check the literal values, joins and filters. If you are sure the answer is "
                        "right, give the same <answer> again."
                    )
                return self._finish("answered" if not out_of_turns else "answered_after_limit")
            if out_of_turns:
                # Prefer an earlier answer that ran (e.g. the empty one before a failed retry),
                # then the last draft that ran; keep the failing query only if there is neither.
                self._finish("failed")
                if self.answer_sql is None:
                    self.answer_sql, self.answer_result = sql, result
                    self.status = "failed"
                return None
            return self._observe(
                f"<result>\nYour answer failed: {result.error}\n</result>\n"
                "Fix the query and give a new <answer>."
            )

        if out_of_turns:
            return self._finish("no_answer")

        if not self.tools:
            return self._finish("no_answer")
        if action.kind == "sql" and action.sql:
            result = self.db.run(action.sql)
            if result.ok and result.rows:
                self._fallback = (action.sql, self.db.run_final(action.sql))
            return self._observe(f"<result>\n{self.db.describe(result)}\n</result>")
        if action.kind == "search":
            return self._observe(f"<result>\n{self.db.search(action.body)}\n</result>")
        return self._observe(FORMAT_REMINDER)


def run_attempts(model: ChatModel, attempts: Sequence[Attempt], *, workers: int = 8) -> int:
    """Advance all attempts to completion, one batched model call per turn. Returns the call count."""
    rounds = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while True:
            active = [a for a in attempts if not a.done]
            if not active:
                return rounds
            rounds += 1
            replies = model.chat([a.messages for a in active], [a.sampling for a in active])

            def apply(pair: tuple[Attempt, ChatResult]) -> None:
                attempt, reply = pair
                try:
                    attempt.step(reply)
                except Exception:  # one broken attempt must not sink the batch
                    log.exception("attempt %s/%d failed", attempt.question_id, attempt.index)
                    attempt._finish("failed")

            list(pool.map(apply, zip(active, replies, strict=True)))
            log.debug("round %d: %d attempts active", rounds, len(active))


def attempt_summary(attempt: Attempt) -> dict[str, Any]:
    result = attempt.answer_result
    return {
        "index": attempt.index,
        "status": attempt.status,
        "turns": attempt.turns,
        "actions": attempt.actions,
        "sql": attempt.answer_sql,
        "ok": bool(result is not None and result.ok),
        "rows": len(result.rows or []) if result is not None and result.ok else None,
        "error": result.error if result is not None and not result.ok else None,
        "key": attempt.key,
    }
