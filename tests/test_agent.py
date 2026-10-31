"""The agent: value lookup, the action protocol, attempts, selection, and a full scored run."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from datalens.agent.chat import ChatResult, ChatSampling, FakeChat
from datalens.agent.config import AgentSpec
from datalens.agent.pipeline import TRACES_FILE, generate_agentic, render_schema
from datalens.agent.prompts import format_rows, selector_prompt
from datalens.agent.protocol import STOP_SEQUENCES, close_turn, parse_action
from datalens.agent.rollout import Attempt, Database, run_attempts
from datalens.agent.select import finish_selection, group_attempts, parse_choice, preliminary
from datalens.agent.values import ValueIndex
from datalens.data.benchmarks import Example, bird_spec, load_examples
from datalens.eval.config import EvalConfig
from datalens.eval.generate import load_benchmark
from datalens.eval.metrics import write_metrics
from datalens.eval.score import score_run
from datalens.sql.executor import execute

# ---------------------------------------------------------------- values


def test_value_index_finds_exact_partial_and_fuzzy_matches(shop_db):
    index = ValueIndex.build(shop_db)
    assert {(m.table, m.column, m.value) for m in index.exact("lamp")} == {("products", "name", "Lamp")}
    assert index.search("Lamp")[0].score == 1.0
    assert index.search("Linus")[0].value == "Linus"
    assert index.search("Lnus")[0].value == "Linus"  # fuzzy: spelling differs
    assert index.search("DE", column="country")[0].ref == "customers.country"
    assert index.search("zzzz") == []


def test_value_hints_name_where_question_phrases_live(shop_db):
    hints = ValueIndex.build(shop_db).hints("What did Grace order besides the Desk?")
    assert any('"Grace" appears in customers.name' in h for h in hints)
    assert any('"Desk" appears in products.name' in h for h in hints)
    assert not any("order" in h.split(" appears")[0].lower() for h in hints)


def test_value_index_cache_roundtrip(shop_db, tmp_path):
    first = ValueIndex.load(shop_db, tmp_path)
    assert list(tmp_path.glob("*.json"))
    second = ValueIndex.load(shop_db, tmp_path)
    assert second.entries == first.entries


# ---------------------------------------------------------------- protocol


@pytest.mark.parametrize(
    ("text", "stop", "kind", "body"),
    [
        ("Let me check.\n<sql>\nSELECT 1\n", "</sql>", "sql", "SELECT 1"),
        ("<search>'Fremont'</search>", None, "search", "Fremont"),
        ("Done.\n<answer>\n```sql\nSELECT name FROM t;\n```\n", "</answer>", "answer", None),
        ("<think>maybe <sql>SELECT 2</sql></think>\n<answer>SELECT 3</answer>", None, "answer", "SELECT 3"),
        ("Here it is:\n```sql\nSELECT 4\n```", None, "answer", None),
        ("I am not sure yet.", None, "none", ""),
        ("<sql>not a query</sql>", None, "sql", "not a query"),
        ("<answer></answer>", None, "none", ""),
    ],
)
def test_parse_action(text, stop, kind, body):
    action = parse_action(text, stop)
    assert action.kind == kind
    if body is not None:
        assert action.body == body
    if kind in ("sql", "answer") and "query" not in text:
        assert action.sql and action.sql.startswith("SELECT")


def test_close_turn_restores_the_stop_tag():
    assert close_turn("<sql>\nSELECT 1", "</sql>").endswith("SELECT 1\n</sql>")
    assert close_turn("<answer>SELECT 1</answer>", None) == "<answer>SELECT 1</answer>"


def test_format_rows_is_compact():
    text = format_rows(["a", "b"], [(1, "x" * 100), (None, b"\x00")], max_rows=1)
    assert "rows (2 total, showing 1)" in text and "..." in text
    assert format_rows(["a"], []) == "columns: a\n(0 rows)"


# ---------------------------------------------------------------- attempts


def _scripted(*turns: str):
    """A responder that plays ``turns`` in order for every conversation."""

    def respond(conversation, sampling):
        assistant_turns = sum(1 for m in conversation if m["role"] == "assistant")
        return turns[min(assistant_turns, len(turns) - 1)]

    return respond


def _attempt(shop_db, **kwargs) -> Attempt:
    db = Database(shop_db, ValueIndex.build(shop_db))
    return Attempt(
        question_id="q",
        index=0,
        messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        sampling=ChatSampling(stop=STOP_SEQUENCES),
        db=db,
        **kwargs,
    )


def test_attempt_searches_explores_and_answers(shop_db):
    attempt = _attempt(shop_db)
    chat = FakeChat(
        _scripted(
            "<search>germany</search>",
            "<sql>SELECT DISTINCT country FROM customers</sql>",
            "<answer>SELECT name FROM customers WHERE country = 'DE'</answer>",
        )
    )
    run_attempts(chat, [attempt])
    assert attempt.status == "answered"
    assert attempt.actions == ["search", "sql", "answer"]
    assert sorted(attempt.answer_result.rows) == [("Ada",), ("Linus",)]
    observations = [m["content"] for m in attempt.transcript if m["role"] == "user"]
    assert "No text values similar" in observations[0]  # 'germany' isn't stored; 'DE' is
    assert "columns: country" in observations[1]
    # Each assistant turn is stored with its closing tag.
    assert attempt.transcript[0]["content"].endswith("</search>")


def test_attempt_gets_one_retry_after_an_empty_answer(shop_db):
    attempt = _attempt(shop_db)
    chat = FakeChat(
        _scripted(
            "<answer>SELECT name FROM customers WHERE country = 'Germany'</answer>",
            "<answer>SELECT name FROM customers WHERE country = 'DE'</answer>",
        )
    )
    run_attempts(chat, [attempt])
    assert attempt.status == "answered" and attempt.turns == 2
    assert "returned 0 rows" in attempt.transcript[1]["content"]
    assert len(attempt.answer_result.rows) == 2


def test_attempt_fixes_a_failing_answer(shop_db):
    attempt = _attempt(shop_db)
    chat = FakeChat(
        _scripted("<answer>SELECT nme FROM customers</answer>", "<answer>SELECT name FROM customers</answer>")
    )
    run_attempts(chat, [attempt])
    assert attempt.status == "answered"
    assert "no such column" in attempt.transcript[1]["content"]


def test_attempt_out_of_turns_falls_back_to_last_draft(shop_db):
    attempt = _attempt(shop_db, max_turns=3)
    chat = FakeChat(_scripted("<sql>SELECT name FROM customers</sql>", "Thinking...", "Still thinking."))
    run_attempts(chat, [attempt])
    assert attempt.status == "fallback"
    assert attempt.answer_sql == "SELECT name FROM customers"
    # The model was told when its last turn came.
    assert "last turn" in attempt.transcript[-2]["content"]


def test_attempt_rejects_writes(shop_db):
    attempt = _attempt(shop_db, max_turns=2)
    chat = FakeChat(
        _scripted("<sql>DELETE FROM customers</sql>", "<answer>SELECT COUNT(*) FROM customers</answer>")
    )
    run_attempts(chat, [attempt])
    assert "read-only" in attempt.transcript[1]["content"]
    assert attempt.answer_result.rows == [(3,)]


def test_direct_mode_answers_in_one_turn_without_feedback(shop_db):
    attempt = _attempt(shop_db, tools=False, max_turns=1)
    run_attempts(FakeChat(_scripted("<answer>SELECT nme FROM customers</answer>")), [attempt])
    assert attempt.turns == 1 and attempt.done
    assert not attempt.answer_result.ok


# ---------------------------------------------------------------- selection


def _answered(shop_db, sqls: list[str]) -> list[Attempt]:
    db = Database(shop_db)
    attempts = []
    for i, sql in enumerate(sqls):
        a = Attempt(question_id="q", index=i, messages=[], sampling=ChatSampling(), db=db)
        a.answer_sql, a.answer_result, a.done = sql, db.run_final(sql), True
        attempts.append(a)
    return attempts


def test_groups_by_result_not_by_text(shop_db):
    attempts = _answered(
        shop_db,
        [
            "SELECT name FROM customers WHERE country = 'DE'",
            "SELECT c.name FROM customers c WHERE c.country IN ('DE')",
            "SELECT name FROM customers",
            "SELECT nme FROM customers",
        ],
    )
    groups = group_attempts(attempts)
    assert [g.members for g in groups] == [[0, 1], [2]]


def test_consensus_skips_the_selector(shop_db):
    spec = AgentSpec(consensus=0.6)
    same = "SELECT COUNT(*) FROM customers"
    selection = preliminary(_answered(shop_db, [same, same, same, "SELECT 1"]), spec)
    assert selection.method == "consensus" and selection.sql == same


def test_selector_votes_override_a_weak_majority(shop_db):
    spec = AgentSpec(consensus=0.75, selector_samples=3, vote_weight=0.5)
    wrong, right = "SELECT name FROM customers", "SELECT name FROM customers WHERE country = 'DE'"
    attempts = _answered(shop_db, [wrong, wrong, right, "SELECT 1"])
    selection = preliminary(attempts, spec)
    assert selection.method == "selector" and selection.shown == [0, 1, 2]
    # Three judgements with candidates in different orders; each picks the 'right' group (index 1).
    judgements = []
    for order in ([0, 1, 2], [2, 0, 1], [1, 2, 0]):
        letter = "ABC"[order.index(1)]
        judgements.append((order, f"Candidate {letter} filters by country.\n<choice>{letter}</choice>"))
    finish_selection(selection, judgements, len(attempts), spec)
    assert selection.sql == right
    assert [j["choice"] for j in selection.judgements] == [1, 1, 1]


def test_parse_choice_takes_the_last_tag():
    assert parse_choice("<choice>A</choice> ... actually <choice>(b)</choice>", 3) == 1
    assert parse_choice("no idea", 3) is None
    assert parse_choice("<choice>D</choice>", 3) is None


def test_selector_prompt_lists_every_candidate():
    text = selector_prompt("Q?", "", [("A", "SELECT 1", "rows"), ("B", "SELECT 2", "rows")])
    assert "Candidate A:" in text and "Candidate B:" in text and "(A, B)" in text


# ---------------------------------------------------------------- end to end

GOLD = {
    "How many customers are there?": "SELECT COUNT(*) FROM customers",
    "List the names of customers from Germany.": "SELECT name FROM customers WHERE country = 'DE'",
    "What is the total revenue of orders for the product named Lamp?": (
        "SELECT SUM(o.quantity * p.price) FROM orders o JOIN products p ON o.product_id = p.id "
        "WHERE p.name = 'Lamp'"
    ),
    "Which customer placed the most orders?": (
        "SELECT c.name FROM customers c JOIN orders o ON o.customer_id = c.id "
        "GROUP BY c.id ORDER BY COUNT(*) DESC LIMIT 1"
    ),
}


def agent_policy(conversation, sampling):
    """Explores once, then answers. Attempts disagree on Germany so the selector has to judge."""
    if conversation[0]["content"].startswith("You are an expert data analyst reviewing"):
        # Selector: pick the candidate that filters on country.
        text = conversation[-1]["content"]
        labels = re.findall(r"Candidate ([A-H]):\n```sql\n(.*?)\n```", text, re.DOTALL)
        best = next((label for label, sql in labels if "country" in sql), labels[0][0])
        return f"<choice>{best}</choice>"
    question = conversation[1]["content"].split("Question: ", 1)[1].strip()
    turns = sum(1 for m in conversation if m["role"] == "assistant")
    if turns == 0:
        return "<sql>SELECT * FROM customers LIMIT 3</sql>"
    if question.startswith("List the names") and not sampling.temperature:
        return "<answer>SELECT name FROM customers</answer>"  # the greedy attempt is wrong
    return f"<answer>{GOLD[question]}</answer>"


def _agent_config(**agent) -> EvalConfig:
    return EvalConfig.model_validate(
        {
            "run_name": "agent-test",
            "model": {"name": "Fake agent", "backend": "openai", "model": "fake", "group": "agentic"},
            "decoding": {"samples": 0},
            "pricing": {"gpu_hourly_usd": 1.0},
            "schema_cache_dir": None,
            "agent": {"rollouts": 4, "consensus": 0.9, "chunk_size": 2, **agent},
        }
    )


def test_agent_run_is_scored_by_the_standard_scorer(bird_root, tmp_path):
    config = _agent_config()
    examples = load_examples(bird_spec(bird_root, "dev"))
    chat = FakeChat(agent_policy)
    run_dir = generate_agentic(config, tmp_path / "runs", model=chat, examples=examples, value_cache_dir=None)

    rows = [json.loads(line) for line in (run_dir / "generations.jsonl").read_text().splitlines()]
    assert len(rows) == 4
    germany = next(r for r in rows if r["id"].endswith("-1"))
    assert germany["agent"]["method"] == "selector"
    assert len(germany["samples"]["completions"]) == 4
    traces = [json.loads(line) for line in (run_dir / TRACES_FILE).read_text().splitlines()]
    assert all(len(t["attempts"]) == 4 for t in traces)
    assert chat.calls[:2] == [8, 8]  # two questions x four attempts per lockstep turn

    score_run(run_dir, examples=examples)
    summary = write_metrics(run_dir)
    assert summary["greedy"]["ex"] == 1.0
    assert summary["sampled"]["num_samples"] == 4

    # Resuming does nothing once every question is answered.
    calls = len(chat.calls)
    generate_agentic(config, tmp_path / "runs", model=chat, examples=examples, value_cache_dir=None)
    assert len(chat.calls) == calls


def test_render_schema_degrades_to_fit(bird_root):
    example = load_examples(bird_spec(bird_root, "dev"))[0]
    full = render_schema(example, AgentSpec(), None)
    assert "ISO country code" in full  # BIRD description
    tight = render_schema(example, AgentSpec(max_schema_chars=len(full) - 1), None)
    assert len(tight) < len(full)


def test_sampled_benchmark_is_fixed_and_can_drop_bad_gold(bird_root, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base = {
        "run_name": "s",
        "model": {"name": "s", "backend": "openai", "model": "fake", "group": "base"},
        "data_root": str(bird_root),
    }
    a = load_benchmark(EvalConfig.model_validate({**base, "sample": 2, "sample_seed": 1}))
    b = load_benchmark(EvalConfig.model_validate({**base, "sample": 2, "sample_seed": 1}))
    assert [e.id for e in a] == [e.id for e in b] and len(a) == 2
    clean = load_benchmark(EvalConfig.model_validate({**base, "drop_bad_gold": True}))
    assert all(execute(e.db_path, e.gold_sql).rows for e in clean)


def test_fake_chat_honours_stop_sequences():
    chat = FakeChat(lambda c, s: "<sql>SELECT 1</sql> and then I imagine a result")
    [result] = chat.chat([[{"role": "user", "content": "x"}]], [ChatSampling(stop=STOP_SEQUENCES)])
    assert isinstance(result, ChatResult)
    assert result.text == "<sql>SELECT 1" and result.stop_sequence == "</sql>"


# ---------------------------------------------------------------- suites and review


def _vllm_config(name: str, model: str = "m", agent: dict | None = None, **kwargs) -> EvalConfig:
    raw = {
        "run_name": name,
        "model": {
            "name": name,
            "backend": "vllm",
            "model": model,
            "group": "agentic" if agent else "frontier",
            "backend_kwargs": {"max_model_len": 32768, **kwargs},
        },
    }
    if agent is not None:
        raw["agent"] = agent
    return EvalConfig.model_validate(raw)


def test_suite_groups_consecutive_runs_that_share_an_engine():
    from datalens.eval.suite import groups_of

    configs = [
        _vllm_config("base"),
        _vllm_config("agent", agent={"chat_kwargs": {"chat_template_kwargs": {"enable_thinking": False}}}),
        _vllm_config("other-model", model="m2"),
        _vllm_config("other-settings", model="m2", quantization="fp8"),
    ]
    assert groups_of(configs) == [[0, 1], [2], [3]]


def test_suite_scores_reviews_and_reports_finished_runs(bird_root, tmp_path, monkeypatch):
    from datalens.eval.generate import generate
    from datalens.eval.suite import run_suite
    from datalens.inference.backends import FakeBackend

    monkeypatch.chdir(tmp_path)
    runs = tmp_path / "runs"
    examples = load_examples(bird_spec(bird_root, "dev"))

    agent_cfg = _agent_config()
    agent_cfg.data_root = str(bird_root)
    generate_agentic(agent_cfg, runs, model=FakeChat(agent_policy), examples=examples, value_cache_dir=None)

    base_cfg = EvalConfig.model_validate(
        {
            "run_name": "base-test",
            "model": {"name": "Fake base", "backend": "openai", "model": "fake", "group": "frontier"},
            "data_root": str(bird_root),
            "pricing": {"gpu_hourly_usd": 1.0},
            "schema_cache_dir": None,
        }
    )
    generate(base_cfg, runs, backend=FakeBackend(lambda p, i: "```sql\nSELECT name FROM customers\n```"))

    for cfg in (agent_cfg, base_cfg):
        (tmp_path / f"{cfg.run_name}.yaml").write_text(cfg.to_yaml())
    (tmp_path / "suite.yaml").write_text(
        "runs:\n  - config: agent-test.yaml\n  - config: base-test.yaml\nreports:\n  reports/out: []\n"
    )
    run_dirs = run_suite(tmp_path / "suite.yaml", runs, isolate=False)
    assert [d.name for d in run_dirs] == ["agent-test", "base-test"]
    assert (tmp_path / "reports/out/results.md").is_file()

    review = [json.loads(line) for line in (runs / "base-test/review.jsonl").read_text().splitlines()]
    wrong = [r for r in review if not r["correct"]]
    assert wrong and all("gold_result" in r and "pred_result" in r for r in wrong)
    agent_review = [json.loads(line) for line in (runs / "agent-test/review.jsonl").read_text().splitlines()]
    germany = next(r for r in agent_review if r["id"].endswith("-1"))
    assert germany["agent"]["method"] == "selector"
    assert [g["correct"] for g in germany["agent"]["groups"]] == [True, False]


# ---------------------------------------------------------------- worked examples


def _ex(i: int, db: str, question: str, sql: str, evidence: str = "") -> Example:
    return Example(
        id=f"t-{i}", db_id=db, question=question, evidence=evidence, gold_sql=sql, db_path=Path("x")
    )


def test_fewshot_retrieves_similar_questions_from_other_databases():
    from datalens.agent.fewshot import FewShotIndex, format_shots

    pool = [
        _ex(0, "shop", "What percentage of orders were shipped late?", "SELECT 1"),
        _ex(
            1,
            "cars",
            "What percentage of cars are red?",
            "SELECT CAST(SUM(color = 'red') AS REAL) * 100 / COUNT(*)",
        ),
        _ex(2, "zoo", "Which animal is the oldest?", "SELECT name FROM animals ORDER BY age DESC LIMIT 1"),
        _ex(
            3,
            "zoo2",
            "What percentage of animals are red pandas?",
            "SELECT CAST(SUM(x) AS REAL) * 100 / COUNT(*)",
        ),
    ]
    index = FewShotIndex(pool)
    shots = index.query("What percentage of customers are from Germany?", "", k=2, exclude_db="shop")
    assert [s.db_id for s in shots] == ["cars", "zoo2"]
    text = format_shots(shots)
    assert "Solved examples" in text and "CAST(SUM(color = 'red') AS REAL)" in text
    assert index.query("anything", "", k=0) == []


def test_agent_prompt_includes_examples_when_enabled(bird_root):
    from datalens.agent.pipeline import _Resources, build_attempts

    examples = load_examples(bird_spec(bird_root, "dev"))
    spec = AgentSpec(fewshot=2, fewshot_benchmark="bird-dev")
    res = _Resources(spec, None, None, str(bird_root))
    attempts, _, _ = build_attempts(examples[0], spec, res)
    prompt = attempts[0].messages[1]["content"]
    # The pool is the same database here, so nothing qualifies: examples are never from the question's db.
    assert "Solved examples" not in prompt
    assert "Question: How many customers are there?" in prompt
