"""Question to SQL to rows, with execution-based self-consistency.

The serving path uses exactly the inference recipe that is evaluated: the same
prompt, ``n`` sampled queries, read-only execution of each, and a majority vote
over their results.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from datalens.data.schema import load_schema
from datalens.inference.backends import Backend, SamplingConfig, create_backend
from datalens.inference.voting import group_candidates
from datalens.prompts import PromptConfig, build_prompt
from datalens.serving.settings import Settings
from datalens.sql.compare import result_key
from datalens.sql.executor import ErrorKind, ExecResult, execute
from datalens.sql.parse import extract_sql, is_read_only

READ_ONLY_ERROR = "only a single read-only SELECT query is allowed"


@dataclass
class Candidate:
    sql: str | None
    votes: int
    ok: bool
    error: str | None
    chosen: bool = False


@dataclass
class Answer:
    sql: str | None
    result: ExecResult | None
    candidates: list[Candidate]
    model: str
    latency_ms: float


class Engine:
    def __init__(
        self,
        backend: Backend,
        *,
        prompt_config: PromptConfig | None = None,
        temperature: float | None = 0.8,
        max_tokens: int = 1024,
        max_rows: int = 500,
        timeout_s: float = 15.0,
    ) -> None:
        self.backend = backend
        self.prompt_config = prompt_config or PromptConfig()
        self.temperature = temperature  # None: the model rejects sampling parameters
        self.max_tokens = max_tokens
        self.max_rows = max_rows
        self.timeout_s = timeout_s

    @classmethod
    def from_settings(cls, settings: Settings) -> Engine:
        kwargs: dict[str, object] = {}
        if settings.backend == "openai":
            kwargs = {"base_url": settings.base_url, "api_key": settings.api_key}
            temperature: float | None = 0.8
            max_tokens = 1024  # the GRPO models were trained with 1,024 completion tokens
        else:
            # Current Claude models reject sampling parameters, and their thinking
            # counts toward max_tokens.
            if settings.effort:
                kwargs = {"effort": settings.effort}
            temperature, max_tokens = None, 16_000
        backend = create_backend(settings.backend, settings.model, **kwargs)
        return cls(
            backend,
            prompt_config=PromptConfig(
                reasoning=settings.reasoning,
                num_examples=settings.num_examples,
                schema_cache_dir=settings.schema_cache_dir,
            ),
            temperature=settings.temperature if settings.temperature is not None else temperature,
            max_tokens=settings.max_tokens or max_tokens,
            max_rows=settings.max_rows,
            timeout_s=settings.query_timeout_s,
        )

    def _execute(self, db_path: Path, sql: str) -> ExecResult:
        if not is_read_only(sql):
            return ExecResult(ok=False, error=READ_ONLY_ERROR, error_kind=ErrorKind.RUNTIME)
        return execute(db_path, sql, timeout_s=self.timeout_s, max_rows=self.max_rows)

    def answer(self, db_path: str | Path, question: str, *, evidence: str = "", samples: int = 1) -> Answer:
        start = time.perf_counter()
        db_path = Path(db_path)
        cfg = self.prompt_config
        schema = load_schema(db_path, num_examples=cfg.num_examples, cache_dir=cfg.schema_cache_dir)
        prompt = build_prompt(
            schema.render(examples=cfg.num_examples > 0, descriptions=cfg.descriptions),
            question,
            evidence,
            reasoning=cfg.reasoning,
        )
        samples = max(1, samples)
        sampling = SamplingConfig(
            n=samples,
            temperature=self.temperature if self.temperature is None or samples > 1 else 0.0,
            max_tokens=self.max_tokens,
        )
        generation = self.backend.generate([prompt], sampling)[0]

        sqls = [extract_sql(c.text) for c in generation.completions]
        results = [self._execute(db_path, sql) if sql else None for sql in sqls]
        keys = [
            result_key(r.rows) if r is not None and r.ok and r.rows is not None else None for r in results
        ]
        empty = [bool(r is not None and r.ok and not r.rows) for r in results]
        groups = group_candidates(keys, empty)

        candidates: list[Candidate] = []
        chosen_index: int | None = groups[0].representative if groups else None
        for group in groups:
            i = group.representative
            candidates.append(
                Candidate(sql=sqls[i], votes=group.votes, ok=True, error=None, chosen=i == chosen_index)
            )
        seen_failed: set[str | None] = set()
        for sql, result in zip(sqls, results, strict=True):
            if result is not None and result.ok:
                continue
            if sql in seen_failed:
                continue
            seen_failed.add(sql)
            error = result.error if result is not None else "no SQL found in the model output"
            candidates.append(Candidate(sql=sql, votes=0, ok=False, error=error))

        if chosen_index is None:
            # Nothing executed: surface the first attempt and its error.
            first = next((i for i, s in enumerate(sqls) if s), None)
            chosen_sql = sqls[first] if first is not None else None
            chosen_result = results[first] if first is not None else None
        else:
            chosen_sql, chosen_result = sqls[chosen_index], results[chosen_index]

        return Answer(
            sql=chosen_sql,
            result=chosen_result,
            candidates=candidates,
            model=self.backend.name,
            latency_ms=1000 * (time.perf_counter() - start),
        )
