"""The agent pipeline end to end on the shop database, with a scripted model in every role."""

from __future__ import annotations

import json

import pytest
import yaml
from typer.testing import CliRunner

from agent_fakes import BROKEN, GOLD, WRONG, Script, agent_config, make_agent_bird
from datalens import cli
from datalens.agent.analysis import analyze
from datalens.agent.models import ModelPool
from datalens.agent.pipeline import STAGE_FILES, AgentRun, read_jsonl
from datalens.agent.selector_data import build_selector_records
from datalens.eval.config import EvalConfig
from datalens.eval.metrics import write_metrics
from datalens.eval.score import score_run
from datalens.inference.backends import FakeBackend

EMPTY = "SELECT name FROM customers WHERE country = 'FR'"


@pytest.fixture
def bird(tmp_path):
    return make_agent_bird(tmp_path / "bird")


def _run(bird, tmp_path, script: Script, stages=None, **overrides):
    config = agent_config(bird, tmp_path, **overrides)
    fake = FakeBackend(script)
    run = AgentRun(config, tmp_path / "runs", pool=ModelPool(config.models, backends={"fake": fake}))
    return run, run.run(stages), fake


def _score(run_dir):
    score_run(run_dir, workers=2)
    return write_metrics(run_dir, n_boot=50)


def test_pipeline_answers_and_scores_like_an_eval_run(bird, tmp_path):
    script = Script()
    _, run_dir, _ = _run(bird, tmp_path, script)
    for name in STAGE_FILES.values():
        assert len(read_jsonl(run_dir / name)) == 4, name
    rows = [json.loads(line) for line in (run_dir / "generations.jsonl").read_text().splitlines()]
    assert len(rows) == 4
    assert all(len(r["samples"]["completions"]) == 4 + 3 for r in rows)  # 4 reasoning + 1 + 2 ICL slots
    config = EvalConfig.from_yaml(run_dir / "config.yaml")
    assert config.model.name == "Fake 3B + agent" and config.model.group == "fine-tuned"

    summary = _score(run_dir)
    assert summary["greedy"]["ex"] == 1.0
    assert summary["sampled"]["pass_at_k"]["7"] == 1.0
    assert summary["greedy"]["cost"]["usd_per_1k_queries"] is not None

    understanding = read_jsonl(run_dir / "understanding.jsonl")
    lamp = understanding["bird-dev-2"]["understanding"]
    assert lamp["source"] == "llm" and lamp["literals"] == ["Lamp"]
    assert ("products", "name", "Lamp") in {(m["table"], m["column"], m["value"]) for m in lamp["matches"]}
    assert [d["id"] for d in lamp["demos"]][:1] == ["bird-train-2"]  # same skeleton, other database
    germany = understanding["bird-dev-1"]["understanding"]
    assert ("customers", "country", "DE") in {
        (m["table"], m["column"], m["value"]) for m in germany["matches"]
    }
    assert (
        script.calls["understanding"] == 8
    )  # the 4 train questions for the example pool, then the 4 dev ones


def test_rerun_resumes_without_calling_the_model(bird, tmp_path):
    _, run_dir, fake = _run(bird, tmp_path, Script())
    calls = len(fake.calls)
    config = agent_config(bird, tmp_path)
    again = AgentRun(config, tmp_path / "runs", pool=ModelPool(config.models, backends={"fake": fake}))
    assert again.run() == run_dir
    assert len(fake.calls) == calls


def test_tournament_rescues_a_minority_answer(bird, tmp_path):
    script = Script(reasoning=lambda q, i: WRONG, icl=lambda q, i: GOLD[q] if i == 0 else WRONG)
    _, run_dir, _ = _run(bird, tmp_path, script)
    summary = _score(run_dir)
    assert summary["greedy"]["ex"] == 1.0  # the selector picks the 2-vote correct group
    assert summary["sampled"]["ex_self_consistency"] == 0.0  # the vote picks the 5-vote wrong one

    result = analyze(run_dir)
    total = {name: row["total"] for name, row in result["ablations"].items()}
    assert total == {
        "Agentar-Scale (full)": 1.0,
        "w/o reasoning generator": 1.0,
        "w/o ICL generator": 0.0,
        "w/o iterative refinement": 1.0,
        "w/o selection scaling (self-consistency)": 0.0,
    }
    assert result["selector"]["decisive_accuracy"] == 1.0
    assert result["overlap"] == {"both": 0, "icl_only": 4, "reasoning_only": 0, "neither": 0}
    assert result["official"] == {"n": 4, "official_ex": 1.0, "disagreements": 0}
    assert "| w/o selection scaling (self-consistency) |" in (run_dir / "analysis.md").read_text()


def test_fixer_repairs_errors_over_several_rounds(bird, tmp_path):
    still_broken = "SELECT nme2 FROM customers"

    def fixer(q, sql):
        return {BROKEN: still_broken, still_broken: GOLD[q]}.get(sql, sql)

    script = Script(reasoning=lambda q, i: BROKEN, icl=lambda q, i: EMPTY, fixer=fixer)
    _, run_dir, _ = _run(bird, tmp_path, script)
    refined = read_jsonl(run_dir / "refinement.jsonl")["bird-dev-0"]["candidates"]
    assert [(c["generator"], c["sql"]) for c in refined] == [
        ("fixer", still_broken),
        ("fixer", GOLD["How many customers are there?"]),
    ]
    assert refined[1]["parent"] == 4 + 3  # the second fix rewrites the first one
    assert refined[1]["root"] == "reasoning"
    assert _score(run_dir)["greedy"]["ex"] == 1.0


def test_reviser_adds_a_candidate_when_it_changes_the_query(bird, tmp_path):
    script = Script(reasoning=lambda q, i: WRONG, icl=lambda q, i: WRONG, revisor=lambda q, sql: GOLD[q])
    _, run_dir, _ = _run(bird, tmp_path, script)
    refined = read_jsonl(run_dir / "refinement.jsonl")["bird-dev-1"]["candidates"]
    assert [c["generator"] for c in refined] == ["revisor"]
    assert _score(run_dir)["greedy"]["ex"] == 1.0
    result = analyze(run_dir)
    assert result["ablations"]["w/o iterative refinement"]["total"] == 0.0


def test_stages_run_one_at_a_time(bird, tmp_path):
    run, run_dir, _ = _run(bird, tmp_path, Script(), stages=["understand"])
    assert (run_dir / "understanding.jsonl").is_file() and not (run_dir / "reasoning.jsonl").exists()
    assert (run_dir / "generations.jsonl").read_text() == ""  # nothing has finished selection yet
    run.run(["reason", "icl", "refine", "select"])
    assert _score(run_dir)["greedy"]["ex"] == 1.0
    with pytest.raises(ValueError, match="unknown stage"):
        run.run(["generate"])


def test_disabled_generator_and_vote_selection(bird, tmp_path):
    script = Script()
    _, run_dir, _ = _run(bird, tmp_path, script, icl={"enabled": False}, selection={"method": "vote"})
    assert not (run_dir / "icl.jsonl").exists()
    assert script.calls["icl"] == 0 and script.calls["selector"] == 0
    rows = [json.loads(line) for line in (run_dir / "generations.jsonl").read_text().splitlines()]
    assert all(len(r["samples"]["completions"]) == 4 for r in rows)
    assert _score(run_dir)["greedy"]["ex"] == 1.0


def test_prompts_longer_than_the_context_are_skipped(bird, tmp_path):
    _, run_dir, _ = _run(
        bird, tmp_path, Script(), max_model_len=600, understanding={"llm": False, "examples": 0}
    )
    reasoning = read_jsonl(run_dir / "reasoning.jsonl")
    assert all(r["candidates"] == [] and r["notes"] for r in reasoning.values())
    rows = [json.loads(line) for line in (run_dir / "generations.jsonl").read_text().splitlines()]
    assert len(rows) == 4  # every question still gets a (possibly empty) answer


def test_unreadable_understanding_falls_back_to_heuristics(bird, tmp_path):
    _, run_dir, _ = _run(bird, tmp_path, Script(json_ok=False))
    u = read_jsonl(run_dir / "understanding.jsonl")["bird-dev-1"]["understanding"]
    assert u["source"] == "heuristic" and u["skeleton"] == "List the names of customers from <VALUE>."
    assert any(m["value"] == "DE" for m in u["matches"])  # n-gram matching needs no model


def test_example_pool_is_built_once_and_reused(bird, tmp_path):
    script = Script()
    _run(bird, tmp_path, script)
    stores = list((tmp_path / "agent-index" / "examples").iterdir())
    assert [s.name for s in stores] == ["bird-train-llm-tfidf"]
    before = script.calls["understanding"]
    _run(bird, tmp_path, script, run_name="second")
    assert script.calls["understanding"] == before  # cached extractions, cached store


def test_selector_data_pairs_a_right_and_a_wrong_query(bird, tmp_path):
    script = Script(icl=lambda q, i: WRONG)
    _, run_dir, _ = _run(
        bird,
        tmp_path,
        script,
        stages=["understand", "reason", "icl", "refine"],
        benchmark="bird-train",
        run_name="train-run",
        selection={"method": "vote"},
    )
    understanding = read_jsonl(run_dir / "understanding.jsonl")
    assert all(not r["understanding"]["demos"] for r in understanding.values())  # same database only
    records = build_selector_records(run_dir, pairs_per_question=2)
    assert records and all(r["label"] in ("1", "2") for r in records)
    for record in records:
        user = record["prompt"][1]["content"]
        gold = GOLD[user.split("# Question:\n", 1)[1].split("\n", 1)[0]]
        chosen = user.split(f"## Candidate {record['label']}\n```sql\n", 1)[1].split("\n```", 1)[0]
        assert " ".join(chosen.split()) == " ".join(gold.split())
    assert record["prompt"][0]["role"] == "system"


def test_cli_runs_scores_and_analyzes(bird, tmp_path, monkeypatch):
    import datalens.agent.models as models

    script = Script(reasoning=lambda q, i: WRONG, icl=lambda q, i: GOLD[q] if i == 0 else WRONG)
    monkeypatch.setattr(models, "create_backend", lambda *a, **k: FakeBackend(script))
    config = agent_config(bird, tmp_path)
    path = tmp_path / "agent.yaml"
    path.write_text(yaml.safe_dump(config.model_dump(mode="json")), encoding="utf-8")
    runner = CliRunner()
    runs = tmp_path / "cli-runs"

    result = runner.invoke(cli.app, ["agent", "index", str(path)])
    assert result.exit_code == 0, result.output
    assert "shop:" in result.output and "example store: 4 bird-train questions" in result.output

    result = runner.invoke(cli.app, ["agent", "run", str(path), "--runs", str(runs), "--stage", "understand"])
    assert result.exit_code == 0, result.output
    result = runner.invoke(cli.app, ["agent", "run", str(path), "--runs", str(runs)])
    assert result.exit_code == 0, result.output
    run_dir = runs / "agent-test"
    result = runner.invoke(cli.app, ["eval", "score", str(run_dir)])
    assert result.exit_code == 0 and "EX 100.0%" in result.output, result.output
    result = runner.invoke(
        cli.app,
        ["agent", "analyze", str(run_dir), "--compare", f"rerun={run_dir}", "--baseline", str(run_dir)],
    )
    assert result.exit_code == 0, result.output
    assert "## Ablations (paper Table 2)" in result.output and "rerun (separate run)" in result.output
    # Against itself: the agent's answer matches its own greedy slot, and beats the 0% vote.
    assert "| greedy | 4 | 100.00 | 100.00 | +0.0" in result.output
    assert "| self-consistency | 4 | 100.00 | 0.00 | +100.0" in result.output

    result = runner.invoke(
        cli.app,
        [
            "agent",
            "selector-data",
            str(run_dir),
            "--out",
            str(tmp_path / "pairs.jsonl"),
            "--max-records",
            "3",
        ],
    )
    assert result.exit_code == 0, result.output
    assert len((tmp_path / "pairs.jsonl").read_text().splitlines()) == 3


def test_a_missing_local_adapter_fails_with_a_hint(bird, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "checkpoints").mkdir()
    selector = {"model": "fake", "lora": "checkpoints/grpo-selector-qwen2.5-coder-3b"}
    with pytest.raises(FileNotFoundError, match="train it with `datalens train grpo"):
        _run(
            bird,
            tmp_path,
            Script(),
            stages=["select"],
            models={"fake": {"backend": "vllm", "model": "fake"}},  # adapters need a vLLM model
            roles={"selector": selector},
        )
