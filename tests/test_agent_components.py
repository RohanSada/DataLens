"""The agent's building blocks: schemas, value matching, examples, prompts, selection logic."""

from __future__ import annotations

import json

import numpy as np
import pytest

from datalens.agent.config import AgentConfig
from datalens.agent.examples import Demo, ExampleStore, TfidfIndex, heuristic_skeleton
from datalens.agent.execution import EMPTY_RESULT, Outcome, QueryRunner, render_rows, run_query
from datalens.agent.models import ModelPool, RoleModel
from datalens.agent.prompts import (
    ICL_STYLES,
    SELECTOR_SYSTEM,
    ShownCandidate,
    fixer_prompt,
    icl_prompt,
    selector_prompt,
    understanding_prompt,
)
from datalens.agent.selection import (
    Candidate,
    Selection,
    choose,
    group_by_result,
    parse_choice,
    restricted_choice,
    schedule,
    tally,
)
from datalens.agent.understanding import Understanding, parse_understanding
from datalens.agent.values import ValueIndex, find_values, load_value_index, match_ratio, render_matches
from datalens.data.schema import load_schema
from datalens.inference.backends import FakeBackend, SamplingConfig
from datalens.prompts import PromptParts
from datalens.rewards import SelectionReward

# ------------------------------------------------------------------ schema rendering


def test_matched_values_come_first_and_other_columns_are_unchanged(shop_db):
    schema = load_schema(shop_db, num_examples=3)
    plain = schema.render()
    assert schema.render(matched=None) == plain == schema.render(matched={})
    matched = schema.render(matched={"customers.country": ["XX", "DE"], "products.name": ["Lamp"]})
    country = next(line for line in matched.splitlines() if line.strip().startswith("country"))
    assert country.endswith('example: ["XX", "DE", "US"]')  # matched first, sampled ones after, no repeats
    names = next(
        line for line in matched.splitlines() if "products" in line or line.strip().startswith("name")
    )
    assert names  # products.name now lists Lamp first
    assert matched.count("CREATE TABLE") == plain.count("CREATE TABLE")
    untouched = [line for line in plain.splitlines() if "orders" in line or "quantity" in line]
    assert all(line in matched.splitlines() for line in untouched)


def test_matched_values_are_capped(shop_db):
    schema = load_schema(shop_db, num_examples=3)
    text = schema.render(matched={"customers.name": [f"v{i}" for i in range(10)]}, max_matched_examples=4)
    line = next(line for line in text.splitlines() if line.strip().startswith("name TEXT"))
    assert json.loads(line.split("example: ", 1)[1]) == ["v0", "v1", "v2", "v3"]


def test_light_schema_is_markdown_with_keys_and_descriptions(shop_db):
    text = load_schema(shop_db, num_examples=3).render_light(matched={"products.name": ["Lamp"]})
    assert "## Table: customers" in text
    assert "| column_name | column_type | column_description | value_examples |" in text
    assert "ISO country code of the customer; values: DE: Germany; US: United States" in text
    assert "### Primary keys" in text and '["id"]' in text
    # Foreign keys are listed under both tables they connect.
    assert text.count("orders.customer_id = customers.id") == 2
    assert '| name | TEXT |  | ["Lamp", "Desk"] |' in text


# ------------------------------------------------------------------ values


def test_match_ratio_checks_only_down_to_the_floor():
    assert match_ratio("lamp", "the lamp sold", 0.85) == 1.0
    assert match_ratio("chinese grand prix", "the chinese grand pri winner", 0.85) == pytest.approx(17 / 18)
    assert match_ratio("germany", "germ", 0.85) == 0.0


def test_value_index_matches_question_mentions(shop_db):
    index = ValueIndex.build(shop_db)
    assert "Lamp" in index.values and "DE" in index.values
    assert "2024-01-01" in index.values  # dates are strings, not numbers
    hits = index.match_text("Germany refers to country = 'DE'\nWhat did customers in Germany buy?")
    assert ("customers", "country", "DE") in {(h.table, h.column, h.value) for h in hits}
    # A short value must appear as a word, not inside one ("de" in "ordered").
    assert not index.match_text("Which products were ordered?")
    lamp = index.match_text("total revenue of the Lamp product")
    assert [(h.table, h.column, h.value) for h in lamp] == [("products", "name", "Lamp")]
    assert [h.value for h in index.match_text("How many lamps were sold?")] == [
        "Lamp"
    ]  # plural still finds it


def test_literal_matching_tolerates_variants(shop_db):
    index = ValueIndex.build(shop_db)
    hits = index.match_literals(["lamps", "graCE"])
    assert {h.value for h in hits} == {"Lamp", "Grace"}
    found = find_values(index, "How many lamps did Grace buy?", ["lamps"])
    assert [h.value for h in found].count("Lamp") == 1
    assert render_matches(found).splitlines()[0].startswith("customers.name: 'Grace'")


def test_value_index_round_trips_through_its_cache(shop_db, tmp_path):
    index = load_value_index(shop_db, cache_dir=tmp_path)
    files = list((tmp_path / "values").glob("shop-*.pkl"))
    assert len(files) == 1
    loaded = ValueIndex.load(files[0])
    assert loaded.values == index.values
    assert loaded.match_text("the Lamp")[0].value == "Lamp"


# ------------------------------------------------------------------ examples


def test_heuristic_skeleton_masks_literals_and_columns():
    skeleton = heuristic_skeleton(
        "How many orders did Ada Lovelace place in 2024 with a quantity over 3 for 'Lamp'?",
        ["quantity", "order_date"],
    )
    assert (
        skeleton == "How many orders did <VALUE> place in <YEAR> with a <COLUMN> over <NUMBER> for <VALUE>?"
    )


def test_tfidf_ranks_similar_skeletons_first():
    index = TfidfIndex(["How many <COLUMN> in <YEAR>?", "List the <COLUMN> of <VALUE>.", "Average <COLUMN>"])
    scores = index.scores("How many <COLUMN> were there in <YEAR>?")
    assert int(np.argmax(scores)) == 0


def test_example_store_excludes_the_questions_database(tmp_path):
    demos = [
        Demo(id="a", db_id="shop", question="q1", evidence="", sql="SELECT 1", skeleton="How many <COLUMN>?"),
        Demo(id="b", db_id="other", question="q2", evidence="", sql="SELECT 2", skeleton="How many <COLUMN>"),
        Demo(
            id="c", db_id="other", question="q3", evidence="", sql="SELECT 3", skeleton="List every <VALUE>"
        ),
    ]
    store = ExampleStore.build(demos)
    assert [d.id for d in store.search_many(["How many <COLUMN>?"], 2)[0]] == ["a", "b"]
    assert [d.id for d in store.search_many(["How many <COLUMN>?"], 2, exclude_dbs=["shop"])[0]] == ["b", "c"]
    store.save(tmp_path / "store")
    again = ExampleStore.load(tmp_path / "store")
    assert again.search_many(["List every <VALUE>"], 1)[0][0].id == "c"


# ------------------------------------------------------------------ understanding


@pytest.mark.parametrize(
    "text",
    [
        '{"database_literals": ["Japan"], "question_skeleton": "How many <COLUMN> in <COUNTRY>?"}',
        '```json\n{"database_literals": ["Japan"], '
        '"question_skeleton": "How many <COLUMN> in <COUNTRY>?"}\n```',
        '<think>hmm {not json}</think> Here: {"database_literals": "Japan", '
        '"question_skeleton": "How many <COLUMN> in <COUNTRY>?"} done',
    ],
)
def test_understanding_output_is_parsed_leniently(text):
    assert parse_understanding(text) == (["Japan"], "How many <COLUMN> in <COUNTRY>?")


def test_unusable_understanding_output_falls_back():
    assert parse_understanding("no json here") == ([], None)
    assert parse_understanding("Skeleton: List <COLUMN>") == ([], "List <COLUMN>")
    assert parse_understanding('{"database_literals": [1, "", null]}') == (["1"], None)


def test_understanding_round_trips():
    u = Understanding(literals=["Lamp"], skeleton="s", source="llm")
    assert Understanding.from_json(json.loads(json.dumps(u.to_json()))) == u


# ------------------------------------------------------------------ execution


def test_run_query_summarises_rows_errors_and_empty_results(shop_db):
    ok = run_query(shop_db, "SELECT name, country FROM customers ORDER BY id", preview_rows=2)
    assert ok.ok and ok.n_rows == 3 and not ok.empty
    assert ok.describe().splitlines() == [
        "Columns: name | country",
        "Rows: 3",
        "Ada | DE",
        "Grace | US",
        "... (1 more rows)",
    ]
    same = run_query(shop_db, "SELECT name, country FROM customers ORDER BY id DESC")
    assert same.key == ok.key  # BIRD compares result sets, so order does not matter
    empty = run_query(shop_db, "SELECT name FROM customers WHERE country = 'FR'")
    assert empty.ok and empty.empty and empty.describe() == EMPTY_RESULT
    bad = run_query(shop_db, "SELECT nme FROM customers")
    assert (
        not bad.ok and bad.error_kind == "schema_error" and bad.describe().startswith("Error: no such column")
    )
    assert run_query(shop_db, None).error_kind == "no_sql"


def test_truncated_results_never_match_full_ones(shop_db):
    full = run_query(shop_db, "SELECT name FROM customers")
    capped = run_query(shop_db, "SELECT name FROM customers", max_rows=2)
    assert capped.key is not None and capped.key.startswith("truncated:") and capped.key != full.key


def test_render_rows_truncates_cells():
    text = render_rows(["c"], [("x" * 100,), (None,)], max_cell=10)
    assert text.splitlines()[2:] == ["xxxxxxx...", "NULL"]


def test_query_runner_executes_each_query_once(shop_db, monkeypatch):
    import datalens.agent.execution as execution

    seen: list[str] = []
    real = execution.run_query
    monkeypatch.setattr(execution, "run_query", lambda db, sql, **kw: seen.append(sql) or real(db, sql, **kw))
    out = QueryRunner(workers=2).run_many(
        [(shop_db, "SELECT 1"), (shop_db, "SELECT 1"), (shop_db, "SELECT 2")]
    )
    assert [o.preview.splitlines()[-1] for o in out] == ["1", "1", "2"]
    assert sorted(seen) == ["SELECT 1", "SELECT 2"]


# ------------------------------------------------------------------ prompts


def test_prompts_put_the_schema_first_and_carry_the_question():
    tu = understanding_prompt("CREATE TABLE t (a INT);", "How many?", "")
    assert (
        tu.schema_block.startswith("# Database Schema\nCREATE TABLE t")
        and "# Evidence\nNone" in tu.question_block
    )
    assert "question_skeleton" in tu.system
    demo = Demo(
        id="d", db_id="x", question="Count rows", evidence="", sql="SELECT COUNT(*) FROM t", skeleton=""
    )
    systems = set()
    for style in ICL_STYLES:
        p = icl_prompt(style, "## Table: t", [demo], "How many?", "e")
        systems.add(p.system)
        assert "## Example 1\n### Question\nCount rows" in p.question_block
        assert p.question_block.rstrip().endswith("Begin! Take a deep breath and think logically.")
    assert len(systems) == 3 and all("Database Admin Instruction" in s for s in systems)
    fix = fixer_prompt("## Table: t", "How many?", "", "SELECT nme FROM t", "Error: no such column: nme")
    assert "# Original SQL\n```sql\nSELECT nme FROM t\n```" in fix.question_block
    sel = selector_prompt(
        "## Table: t",
        "None",
        "How many?",
        "",
        [ShownCandidate("SELECT 1", "Rows: 1"), ShownCandidate("SELECT 2", "x")],
    )
    assert sel.system == SELECTOR_SYSTEM and "\\boxed{2}" in sel.system
    assert (
        "## Candidate 1\n```sql\nSELECT 1\n```" in sel.question_block
        and "## Candidate 2" in sel.question_block
    )


# ------------------------------------------------------------------ selection


def _cand(key: str | None, generator: str = "reasoning", empty: bool = False) -> Candidate:
    outcome = Outcome(ok=key is not None, key=key, empty=empty, error=None if key else "boom")
    return Candidate(generator=generator, source=generator, sql=f"SELECT '{key}'", outcome=outcome)


def test_groups_rank_by_votes_then_non_empty():
    pool = [_cand("a"), _cand("b", empty=True), _cand("b", empty=True), _cand(None), _cand("c"), _cand("c")]
    groups = group_by_result(pool)
    assert [(g.key, g.members) for g in groups] == [("c", (4, 5)), ("b", (1, 2)), ("a", (0,))]
    assert [g.key for g in group_by_result(pool, allowed={0, 1})] == ["a", "b"]


def test_schedule_is_a_deterministic_round_robin():
    groups = group_by_result([_cand(k) for k in "abcd"])
    matches = schedule(groups, seed="q1")
    assert len(matches) == 6 and matches == schedule(groups, seed="q1")
    assert {frozenset((m.first, m.second)) for m in matches} == {
        frozenset(p) for p in [("a", "b"), ("a", "c"), ("a", "d"), ("b", "c"), ("b", "d"), ("c", "d")]
    }
    assert len(schedule(groups, max_candidates=3)) == 3
    assert len(schedule(groups, both_orders=True)) == 12


def test_most_wins_takes_it_and_ties_go_to_votes():
    pool = [_cand("a"), _cand("a"), _cand("b"), _cand("c")]
    groups = group_by_result(pool)
    matches = schedule(groups, seed="s")
    # c beats everyone, a beats b
    winners = [1 if m.first == "c" or (m.first == "a" and m.second == "b") else 2 for m in matches]
    wins = tally(matches, winners)
    assert wins == {"a": 1.0, "b": 0.0, "c": 2.0}
    assert choose(groups, wins).key == "c"
    assert choose(groups, {}).key == "a"  # no judgments: most votes, as self-consistency would
    assert tally(matches, [None] * len(matches)) == {"a": 0.0, "b": 0.0, "c": 0.0}


def test_restricted_choice_reuses_the_judgments_that_remain():
    pool = [_cand("a"), _cand("a"), _cand("b", "icl"), _cand("c", "icl")]
    groups = group_by_result(pool)
    matches = schedule(groups, seed="s")
    winners = [
        1 if m.first == "c" else 2 if m.second == "c" else (1 if m.first == "b" else 2) for m in matches
    ]
    sel = Selection(method="tournament", chosen=3, vote=0, groups=groups, matches=matches, winners=winners)
    assert restricted_choice(pool, sel, {0, 1, 2, 3}) == (3, 0)
    assert restricted_choice(pool, sel, {0, 1, 2}) == (2, 0)  # without c, b beat a
    assert restricted_choice(pool, sel, {0, 1}) == (0, 0)


@pytest.mark.parametrize(
    ("text", "strict", "expected"),
    [
        ("so the answer is \\boxed{2}", True, 2),
        ("\\boxed{1} ... on reflection \\boxed{2}", True, 2),
        ("\\boxed{Candidate 1}", True, 1),
        ("\\boxed{3}", True, None),
        ("I choose candidate 2.", True, None),
        ("I choose candidate 2.", False, 2),
        ("<think>\\boxed{1}</think> nothing", True, None),
    ],
)
def test_parse_choice(text, strict, expected):
    assert parse_choice(text, 2, strict=strict) == expected


def test_selection_reward_pays_only_for_the_right_boxed_label():
    logged: dict[str, float] = {}
    reward = SelectionReward()
    completions = [
        [{"role": "assistant", "content": "A is right \\boxed{1}"}],
        "\\boxed{2}",
        "candidate 1",
        "\\boxed{2}",
    ]
    scores = reward(completions=completions, label=["1", "1", "1", "2"], log_metric=logged.__setitem__)
    assert scores == [1.0, 0.0, 0.0, 1.0]
    assert logged == {"select/accuracy": 0.5, "select/answered": 0.75}


# ------------------------------------------------------------------ config and models


def _raw_config(**roles):
    return {
        "run_name": "x",
        "name": "x",
        "models": {"m": {"backend": "openai", "model": "fake"}, "v": {"backend": "vllm", "model": "local"}},
        "roles": {
            r: roles.get(r, {"model": "m"})
            for r in ("understanding", "reasoning_generator", "icl_generator", "fixer", "revisor", "selector")
        },
    }


def test_config_checks_role_models():
    AgentConfig.model_validate(_raw_config(selector={"model": "v", "lora": "adapter"}))
    with pytest.raises(ValueError, match="not under `models`"):
        AgentConfig.model_validate(_raw_config(fixer={"model": "missing"}))
    with pytest.raises(ValueError, match="LoRA adapters need a vllm model"):
        AgentConfig.model_validate(_raw_config(fixer={"model": "m", "lora": "adapter"}))


def test_eval_config_describes_the_agent_as_a_fine_tuned_model():
    cfg = AgentConfig.model_validate(_raw_config(reasoning_generator={"model": "v", "lora": "ckpt/grpo"}))
    ev = cfg.eval_config(pool_size=17)
    assert ev.model.group == "fine-tuned" and ev.model.model == "ckpt/grpo" and ev.model.backend == "vllm"
    assert ev.decoding.samples == 17 and ev.prompt.reasoning


class _Closable(FakeBackend):
    closed = False

    def close(self) -> None:
        self.closed = True


def test_pool_loads_lazily_and_releases_unused_models():
    made = []

    def factory(kind, model, **kwargs):
        made.append((kind, model, kwargs))
        return _Closable(lambda p, i: "x", name=model)

    cfg = AgentConfig.model_validate(_raw_config())
    pool = ModelPool(cfg.models, factory=factory)
    assert made == []
    first = pool.backend("m")
    assert pool.backend("m") is first and len(made) == 1
    pool.backend("v")
    pool.release(keep={"v"})
    assert first.closed and pool.loaded() == ["v"]


def test_role_model_passes_its_adapter_and_checks_length():
    seen = {}

    class Lora(FakeBackend):
        def generate(self, prompts, sampling, *, lora_path=None):
            seen["lora"] = lora_path
            return super().generate(prompts, sampling)

        def count_tokens(self, prompt):
            return 900

    role = RoleModel("selector", Lora(lambda p, i: "ok"), lora="ckpt/sel", max_model_len=1000)
    prompt = PromptParts(system="s", schema_block="x" * 3000, question_block="q")
    assert role.generate([prompt], SamplingConfig())[0].completions[0].text == "ok"
    assert seen["lora"] == "ckpt/sel"
    assert role.fits(prompt, 100) and not role.fits(prompt, 101)
    assert RoleModel("x", Lora(lambda p, i: "ok")).fits(prompt, 10**6)  # no limit configured
