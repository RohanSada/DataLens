"""The pipeline's stages, each run on a batch of questions at once.

Batching a whole stage at a time (rather than walking one question through every
step) lets vLLM schedule thousands of prompts together and means each model is
loaded once per stage. The stages, in order:

``understand``  literals and skeleton (model), matched values, few-shot examples
``reason``      candidates from the RL-trained reasoning generator (DDL prompt)
``icl``         candidates from the ICL generator (light schema, examples, 3 styles)
``refine``      fixer for failing / empty results, reviser for the rest, iterated
``select``      pairwise round-robin tournament over result groups
"""

from __future__ import annotations

import json
import logging
import random
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from datalens.agent.config import AgentConfig
from datalens.agent.examples import Demo, ExampleStore, heuristic_skeleton, make_embedder
from datalens.agent.execution import Outcome, QueryRunner
from datalens.agent.models import ModelPool, RoleModel
from datalens.agent.prompts import (
    ShownCandidate,
    fixer_prompt,
    icl_prompt,
    revisor_prompt,
    selector_prompt,
    understanding_prompt,
)
from datalens.agent.selection import (
    Candidate,
    Selection,
    choose,
    group_by_result,
    parse_choice,
    schedule,
    tally,
)
from datalens.agent.understanding import Understanding, parse_understanding
from datalens.agent.values import find_values, load_value_index, render_matches
from datalens.data.benchmarks import Example, load_examples, resolve_spec
from datalens.data.schema import DatabaseSchema, load_schema
from datalens.inference.backends import Generation, SamplingConfig
from datalens.prompts import build_prompt
from datalens.sql.parse import extract_sql

log = logging.getLogger(__name__)

NOT_RUN = Outcome(ok=False, error="not executed", error_kind="not_run")


@dataclass
class Usage:
    """Model calls one question cost in one stage."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0  # amortised GPU time (vLLM) or wall time of the calls (APIs)

    def add(self, gen: Generation) -> None:
        self.calls += 1
        self.input_tokens += gen.input_tokens or 0
        self.output_tokens += sum(c.output_tokens or 0 for c in gen.completions)
        self.latency_s += gen.latency_s

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.calls + other.calls,
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.latency_s + other.latency_s,
        )

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["latency_s"] = round(self.latency_s, 4)
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any] | None) -> Usage:
        return cls(**data) if data else cls()


@dataclass
class Item:
    """One question and everything the stages so far produced for it."""

    example: Example
    understanding: Understanding = field(default_factory=lambda: Understanding(source="disabled"))
    reasoning: list[Candidate] = field(default_factory=list)
    icl: list[Candidate] = field(default_factory=list)
    refined: list[Candidate] = field(default_factory=list)
    selection: Selection | None = None
    usage: Usage = field(default_factory=Usage)  # of the stage being run
    notes: list[str] = field(default_factory=list)
    _light: str | None = field(default=None, repr=False)

    @property
    def id(self) -> str:
        return self.example.id

    def pool(self) -> list[Candidate]:
        """Every candidate, indexed as selection and refinement refer to them."""
        return [*self.reasoning, *self.icl, *self.refined]


class AgentContext:
    """What every stage needs: config, models, the query runner and the indexes."""

    def __init__(self, config: AgentConfig, pool: ModelPool, *, runner: QueryRunner | None = None) -> None:
        self.config = config
        self.pool = pool
        self.runner = runner or QueryRunner(
            timeout_s=config.execution.timeout_s,
            preview_rows=config.execution.preview_rows,
            workers=config.execution.workers,
        )
        self.index_dir = Path(config.index_dir)
        self._store: ExampleStore | None = None

    def role(self, name: str) -> RoleModel:
        spec = self.config.roles.get(name)
        if spec.lora:
            _check_adapter(name, spec.lora)
        model = self.pool.role(name, spec)
        if self.config.max_model_len:
            model.max_model_len = self.config.max_model_len
        return model

    def schema(self, example: Example, num_examples: int = 3) -> DatabaseSchema:
        return load_schema(example.db_path, num_examples=num_examples, cache_dir=self.config.schema_cache_dir)

    def light_schema(self, item: Item) -> str:
        """The light schema with the question's matched values, rendered once per item."""
        if item._light is None:
            schema = self.schema(item.example)
            item._light = schema.render_light(matched=item.understanding.matched())
        return item._light

    def sampling(
        self, n: int, temperature: float | None, max_tokens: int, offset: int = 0, **kw: Any
    ) -> SamplingConfig:
        return SamplingConfig(
            n=n, temperature=temperature, max_tokens=max_tokens, seed=self.config.seed + offset, **kw
        )

    def example_store(self) -> ExampleStore:
        if self._store is None:
            self._store = ensure_example_store(self)
        return self._store


def _check_adapter(role: str, lora: str) -> None:
    """Fail early, and helpfully, when a local adapter folder is missing (e.g. an untrained selector)."""
    path = Path(lora)
    local = path.is_absolute() or (len(path.parts) > 1 and Path(path.parts[0]).is_dir())
    if local and not (path / "adapter_config.json").is_file():
        hint = (
            "train it with `datalens train grpo configs/train/grpo-selector-qwen2.5-coder-3b.yaml`, or use "
            "configs/agent/agentar-qwen2.5-coder-3b-zeroshot.yaml"
            if role == "selector"
            else "check the path, or point the role at a merged model instead"
        )
        raise FileNotFoundError(f"LoRA adapter for role {role} not found at {lora}: {hint}")


def _model_slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")[-120:] or "model"


def _execute(ctx: AgentContext, jobs: Sequence[tuple[Item, Candidate]]) -> None:
    outcomes = ctx.runner.run_many([(it.example.db_path, c.sql) for it, c in jobs])
    for (_, cand), outcome in zip(jobs, outcomes, strict=True):
        cand.outcome = outcome


def _normalise_sql(sql: str) -> str:
    return " ".join(sql.split()).rstrip(";").strip()


# ------------------------------------------------------------------ understand


class UnderstandingCache:
    """Literal and skeleton extractions per question id, appended to a JSONL file.

    The same extraction is needed to build the example pool and to understand
    questions of that split, and a long extraction run must survive a disconnect.
    The file is keyed by the understanding model's name only: delete it after
    changing that model's prompt or sampling settings.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.rows: dict[str, dict[str, Any]] = {}
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.rows[row["id"]] = row

    def add(self, rows: Sequence[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            for row in rows:
                self.rows[row["id"]] = row
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_understanding(
    ctx: AgentContext, examples: Sequence[Example], *, chunk: int = 512
) -> dict[str, dict[str, Any]]:
    """Run the understanding model on ``examples`` (cached); returns rows by id."""
    spec = ctx.config.understanding
    role = ctx.role("understanding")
    cache = UnderstandingCache(ctx.index_dir / "understanding" / f"{_model_slug(role.name)}.jsonl")
    todo = [e for e in examples if e.id not in cache.rows]
    for start in range(0, len(todo), chunk):
        batch = todo[start : start + chunk]
        prompts, owners = [], []
        rows: list[dict[str, Any]] = []
        for ex in batch:
            try:
                schema_text = ctx.schema(ex).render(examples=True)
            except (FileNotFoundError, ValueError):
                rows.append({"id": ex.id, "literals": [], "skeleton": None, "usage": None})
                continue
            prompt = understanding_prompt(schema_text, ex.question, ex.evidence)
            if role.fits(prompt, spec.max_tokens):
                prompts.append(prompt)
                owners.append(ex)
            else:
                rows.append({"id": ex.id, "literals": [], "skeleton": None, "usage": None})
        gens = role.generate(prompts, ctx.sampling(1, spec.temperature, spec.max_tokens))
        for ex, gen in zip(owners, gens, strict=True):
            text = gen.completions[0].text if gen.completions else ""
            literals, skeleton = parse_understanding(text)
            usage = Usage()
            usage.add(gen)
            rows.append(
                {
                    "id": ex.id,
                    "literals": literals,
                    "skeleton": skeleton,
                    "raw": text,
                    "usage": usage.to_json(),
                }
            )
        cache.add(rows)
        if len(todo) > chunk:
            log.info("understood %d/%d questions", min(start + chunk, len(todo)), len(todo))
    return {e.id: cache.rows[e.id] for e in examples}


def example_store_dir(config: AgentConfig) -> Path:
    spec = config.understanding
    embedder = _model_slug(spec.embedding_model or "tfidf")
    return Path(config.index_dir) / "examples" / f"{spec.example_split}-{spec.example_skeletons}-{embedder}"


def ensure_example_store(ctx: AgentContext) -> ExampleStore:
    """Load the few-shot example pool, building it first if needed (offline preprocessing)."""
    spec = ctx.config.understanding
    directory = example_store_dir(ctx.config)
    embedder = make_embedder(spec.embedding_model)
    if ExampleStore.exists(directory):
        return ExampleStore.load(directory, embedder)

    examples = load_examples(resolve_spec(spec.example_split, ctx.config.data_root))
    log.info("building the example store from %d %s questions", len(examples), spec.example_split)
    llm_rows = extract_understanding(ctx, examples) if spec.example_skeletons == "llm" else {}
    demos = []
    for ex in examples:
        skeleton = (llm_rows.get(ex.id) or {}).get("skeleton")
        if not skeleton:
            try:
                columns: set[str] = ctx.schema(ex).column_names()
            except (FileNotFoundError, ValueError):
                columns = set()
            skeleton = heuristic_skeleton(ex.question, columns)
        demos.append(
            Demo(
                id=ex.id,
                db_id=ex.db_id,
                question=ex.question,
                evidence=ex.evidence,
                sql=ex.gold_sql,
                skeleton=skeleton,
            )
        )
    store = ExampleStore.build(demos, embedder)
    store.save(directory)
    return store


def understand(items: Sequence[Item], ctx: AgentContext) -> None:
    """Step 1: literals, skeleton, matched values and few-shot examples."""
    spec = ctx.config.understanding
    if not spec.enabled:
        for it in items:
            it.understanding = Understanding(source="disabled")
        return
    extracted = extract_understanding(ctx, [it.example for it in items]) if spec.llm else {}
    for it in items:
        ex = it.example
        row = extracted.get(ex.id) or {}
        it.usage += Usage.from_json(row.get("usage"))
        literals = list(row.get("literals") or [])
        skeleton = row.get("skeleton")
        source = "llm" if skeleton else "heuristic"
        if not skeleton:
            skeleton = heuristic_skeleton(ex.question, ctx.schema(ex).column_names())
        matches = []
        if spec.value_retrieval:
            index = load_value_index(ex.db_path, cache_dir=ctx.index_dir)
            text = f"{ex.evidence}\n{ex.question}" if ex.evidence else ex.question
            matches = find_values(
                index,
                text,
                literals,
                threshold=spec.match_threshold,
                limit=spec.max_matched_values,
                literal_threshold=spec.literal_threshold,
                per_literal=spec.per_literal,
            )
        it.understanding = Understanding(literals=literals, skeleton=skeleton, matches=matches, source=source)

    if spec.examples > 0 and ctx.config.icl.enabled:
        store = ctx.example_store()
        # On the example split itself (building selector data on BIRD train), take examples from
        # other databases only, as at test time.
        same_split = ctx.config.benchmark == spec.example_split
        found = store.search_many(
            [it.understanding.skeleton for it in items],
            spec.examples,
            exclude_dbs=[it.example.db_id if same_split else None for it in items],
        )
        for it, demos in zip(items, found, strict=True):
            it.understanding.demos = demos


# ------------------------------------------------------------------ generate


def reason(items: Sequence[Item], ctx: AgentContext) -> None:
    """Step 2a: the RL-trained reasoning generator, with the prompt it was trained on."""
    spec = ctx.config.reasoning
    role = ctx.role("reasoning_generator")
    prompts, owners = [], []
    for it in items:
        ex = it.example
        matched = it.understanding.matched() if spec.matched_values else None
        schema_text = ctx.schema(ex, spec.num_examples).render(
            examples=spec.num_examples > 0, descriptions=False, matched=matched
        )
        prompt = build_prompt(schema_text, ex.question, ex.evidence, reasoning=True)
        if role.fits(prompt, spec.max_tokens):
            prompts.append(prompt)
            owners.append(it)
        else:
            it.notes.append("reasoning prompt longer than the context")
    gens = role.generate(prompts, ctx.sampling(spec.n, spec.temperature, spec.max_tokens, top_p=spec.top_p))
    jobs = []
    for it, gen in zip(owners, gens, strict=True):
        it.usage.add(gen)
        for j, comp in enumerate(gen.completions):
            cand = Candidate(
                generator="reasoning",
                source=f"reasoning#{j}",
                sql=extract_sql(comp.text),
                outcome=NOT_RUN,
                finish_reason=comp.finish_reason,
                output_tokens=comp.output_tokens,
                text=comp.text,
            )
            it.reasoning.append(cand)
            jobs.append((it, cand))
    _execute(ctx, jobs)


def icl(items: Sequence[Item], ctx: AgentContext) -> None:
    """Step 2b: the ICL generator over prompt styles and temperatures, with shuffled examples."""
    spec = ctx.config.icl
    role = ctx.role("icl_generator")
    jobs = []
    for vi, variant in enumerate(spec.variants):
        prompts, owners = [], []
        for it in items:
            ex = it.example
            demos = list(it.understanding.demos)
            if spec.shuffle_examples and len(demos) > 1:
                random.Random(f"{ctx.config.seed}:{ex.id}:{vi}").shuffle(demos)
            prompt = icl_prompt(variant.style, ctx.light_schema(it), demos, ex.question, ex.evidence)
            if role.fits(prompt, spec.max_tokens):
                prompts.append(prompt)
                owners.append(it)
            elif vi == 0:
                it.notes.append("ICL prompt longer than the context")
        gens = role.generate(
            prompts, ctx.sampling(variant.n, variant.temperature, spec.max_tokens, offset=vi)
        )
        label = f"icl:{variant.style}@{variant.temperature}"
        for it, gen in zip(owners, gens, strict=True):
            it.usage.add(gen)
            for k, comp in enumerate(gen.completions):
                cand = Candidate(
                    generator="icl",
                    source=f"{label}#{k}",
                    sql=extract_sql(comp.text),
                    outcome=NOT_RUN,
                    finish_reason=comp.finish_reason,
                    output_tokens=comp.output_tokens,
                    text=comp.text,
                )
                it.icl.append(cand)
                jobs.append((it, cand))
    _execute(ctx, jobs)


# ------------------------------------------------------------------ refine


def _refinement_targets(item: Item, seed: int) -> list[int]:
    """One randomly chosen query per execution outcome (paper section 3.3.3)."""
    groups: dict[str, list[int]] = {}
    for i, cand in enumerate(item.pool()):
        if cand.sql is None or cand.outcome.error_kind == "not_run":
            continue
        if not cand.outcome.ok:
            key = f"error:{cand.outcome.error}"
        elif cand.outcome.empty:
            key = "empty"
        else:
            key = f"rows:{cand.outcome.key}"
        groups.setdefault(key, []).append(i)
    rng = random.Random(f"{seed}:{item.id}:refine")
    return sorted(rng.choice(members) for members in groups.values())


def _needs_fix(cand: Candidate) -> bool:
    return cand.sql is not None and (not cand.outcome.ok or cand.outcome.empty)


def _rewrite(
    ctx: AgentContext, role: RoleModel, kind: str, jobs: Sequence[tuple[Item, int]], temperature: float | None
) -> list[tuple[Item, Candidate]]:
    """Ask the fixer or reviser to rewrite pool entries; returns the new candidates."""
    spec = ctx.config.refinement
    build = fixer_prompt if kind == "fixer" else revisor_prompt
    prompts, owners = [], []
    for it, cid in jobs:
        cand = it.pool()[cid]
        assert cand.sql is not None
        ex = it.example
        prompt = build(ctx.light_schema(it), ex.question, ex.evidence, cand.sql, cand.outcome.describe())
        if role.fits(prompt, spec.max_tokens):
            prompts.append(prompt)
            owners.append((it, cid))
    gens = role.generate(prompts, ctx.sampling(1, temperature, spec.max_tokens, offset=17))
    created = []
    for (it, cid), gen in zip(owners, gens, strict=True):
        it.usage.add(gen)
        comp = gen.completions[0] if gen.completions else None
        sql = extract_sql(comp.text) if comp else None
        if comp is None or sql is None:
            continue
        known = {_normalise_sql(c.sql) for c in it.pool() if c.sql}
        if _normalise_sql(sql) in known:
            continue  # unchanged, or a query the pool already has
        parent = it.pool()[cid]
        cand = Candidate(
            generator=kind,
            source=f"{kind}<{cid}",
            sql=sql,
            outcome=NOT_RUN,
            root=parent.root,
            parent=cid,
            finish_reason=comp.finish_reason,
            output_tokens=comp.output_tokens,
            text=comp.text,
        )
        it.refined.append(cand)
        created.append((it, cand))
    _execute(ctx, created)
    return created


def refine(items: Sequence[Item], ctx: AgentContext) -> None:
    """Step 2c: iterative refinement. The fixer repairs errors and empty results; the
    reviser checks one query per result group for logical mistakes. Fixes that still
    fail go back to the fixer, up to ``max_rounds`` rounds in all."""
    spec = ctx.config.refinement
    fixer = ctx.role("fixer") if spec.fixer else None
    revisor = ctx.role("revisor") if spec.revisor else None
    fix_jobs: list[tuple[Item, int]] = []
    revise_jobs: list[tuple[Item, int]] = []
    for it in items:
        for cid in _refinement_targets(it, ctx.config.seed):
            if _needs_fix(it.pool()[cid]):
                fix_jobs.append((it, cid))
            else:
                revise_jobs.append((it, cid))

    created: list[tuple[Item, Candidate]] = []
    if fixer is not None:
        created += _rewrite(ctx, fixer, "fixer", fix_jobs, spec.fixer_temperature)
    if revisor is not None:
        created += _rewrite(ctx, revisor, "revisor", revise_jobs, spec.revisor_temperature)
    for _ in range(spec.max_rounds - 1):
        if fixer is None:
            break
        again = []
        for it, cand in created:
            if _needs_fix(cand):
                again.append((it, len(it.reasoning) + len(it.icl) + it.refined.index(cand)))
        if not again:
            break
        created = _rewrite(ctx, fixer, "fixer", again, spec.fixer_temperature)


# ------------------------------------------------------------------ select


def select(items: Sequence[Item], ctx: AgentContext) -> None:
    """Step 3: group by result, then a pairwise round-robin judged by the selector."""
    spec = ctx.config.selection
    role = ctx.role("selector") if spec.method == "tournament" else None
    jobs: list[tuple[Item, int, Any]] = []
    for it in items:
        pool = it.pool()
        groups = group_by_result(pool)
        vote = groups[0].representative if groups else None
        sel = Selection(method=spec.method, chosen=vote, vote=vote, groups=groups)
        it.selection = sel
        if not groups:
            sel.note = "no candidate executed"
            continue
        if role is None or len(groups) == 1:
            continue
        sel.matches = schedule(
            groups,
            max_candidates=spec.max_candidates,
            both_orders=spec.both_orders,
            seed=f"{ctx.config.seed}:{it.id}",
        )
        sel.winners = [None] * len(sel.matches)
        sel.texts = [None] * len(sel.matches)
        by_key = {g.key: g for g in groups}
        ex = it.example
        matched = render_matches(it.understanding.matches)
        for mi, match in enumerate(sel.matches):
            shown = []
            for key in (match.first, match.second):
                cand = pool[by_key[key].representative]
                shown.append(ShownCandidate(sql=cand.sql or "", result=cand.outcome.describe()))
            prompt = selector_prompt(ctx.light_schema(it), matched, ex.question, ex.evidence, shown)
            if role.fits(prompt, spec.max_tokens):
                jobs.append((it, mi, prompt))
            else:
                sel.note = "some pairs did not fit the context and were not judged"

    if role is not None and jobs:
        gens = role.generate([p for *_, p in jobs], ctx.sampling(1, spec.temperature, spec.max_tokens))
        for (it, mi, _), gen in zip(jobs, gens, strict=True):
            it.usage.add(gen)
            text = gen.completions[0].text if gen.completions else ""
            assert it.selection is not None
            it.selection.winners[mi] = parse_choice(text, 2, strict=False)
            it.selection.texts[mi] = text

    for it in items:
        done = it.selection
        if done is None or not done.matches:
            continue
        done.wins = tally(done.matches, done.winners)
        entrants = [g for g in done.groups if g.key in done.wins]
        best = choose(entrants, done.wins)
        done.chosen = best.representative if best is not None else done.vote


STAGE_FUNCTIONS = {
    "understand": understand,
    "reason": reason,
    "icl": icl,
    "refine": refine,
    "select": select,
}
