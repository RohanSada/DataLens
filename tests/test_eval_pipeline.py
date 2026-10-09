from __future__ import annotations

import json

import pytest

from datalens.data.benchmarks import bird_spec, load_examples
from datalens.eval import plots
from datalens.eval.config import EvalConfig, PricingSpec
from datalens.eval.generate import generate
from datalens.eval.metrics import query_cost_usd, write_metrics
from datalens.eval.report import build_report, load_runs
from datalens.eval.score import score_run
from datalens.inference.backends import FakeBackend
from datalens.prompts import PromptParts

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


def _question(prompt: PromptParts) -> str:
    return prompt.question_block.split("Question: ", 1)[1].split("\n", 1)[0]


def oracle(prompt: PromptParts, i: int) -> str:
    """Always right."""
    return f"<think>easy</think>\n```sql\n{GOLD[_question(prompt)]}\n```"


def flaky(prompt: PromptParts, i: int) -> str:
    """Greedy (i=0) is wrong on Germany; samples vote their way to the right answer."""
    q = _question(prompt)
    if q.startswith("List the names") and i in (0, 3):
        return "```sql\nSELECT name FROM customers\n```"
    if q.startswith("Which customer") and i == 0:
        return "```sql\nSELECT nme FROM customers\n```"
    return f"```sql\n{GOLD[q]}\n```"


def _config(name: str, group: str, *, samples: int = 0, **pricing) -> EvalConfig:
    return EvalConfig.model_validate(
        {
            "run_name": name,
            "model": {"name": name, "backend": "openai", "model": "fake", "group": group, "params_b": 1.5},
            "decoding": {"samples": samples, "temperature": 0.8},
            "pricing": pricing,
            "schema_cache_dir": None,
        }
    )


def _run(tmp_path, bird_root, name, group, responder, **kwargs):
    examples = load_examples(bird_spec(bird_root, "dev"))
    config = _config(name, group, **kwargs)
    run_dir = generate(
        config, tmp_path / "runs", backend=FakeBackend(responder, latency_s=0.05), examples=examples
    )
    score_run(run_dir, examples=examples, workers=2, ves_iterations=2)
    return run_dir, write_metrics(run_dir, n_boot=200)


def test_generate_is_resumable(tmp_path, bird_root):
    examples = load_examples(bird_spec(bird_root, "dev"))
    config = _config("resume", "fine-tuned")
    backend = FakeBackend(oracle)
    generate(config, tmp_path / "runs", backend=backend, examples=examples[:2])
    generate(config, tmp_path / "runs", backend=backend, examples=examples)
    lines = (tmp_path / "runs" / "resume" / "generations.jsonl").read_text().splitlines()
    assert len(lines) == 4
    assert len({json.loads(line)["id"] for line in lines}) == 4


def test_oracle_scores_perfectly(tmp_path, bird_root):
    _, summary = _run(
        tmp_path, bird_root, "oracle", "frontier", oracle, input_per_mtok=2.0, output_per_mtok=10.0
    )
    greedy = summary["greedy"]
    assert greedy["ex"] == 1.0
    assert greedy["errors"]["correct"] == 4
    assert greedy["soft_f1"] == 1.0
    assert greedy["r_ves"] is not None and greedy["r_ves"] > 0
    assert greedy["schema_linking"]["table_recall"] == 1.0
    assert set(greedy["ex_by_difficulty"]) == {"simple", "moderate", "challenging"}
    assert greedy["cost"]["usd_per_1k_queries"] > 0


def test_self_consistency_recovers_from_bad_greedy(tmp_path, bird_root):
    _, summary = _run(tmp_path, bird_root, "small", "fine-tuned", flaky, samples=5, gpu_hourly_usd=2.0)
    assert summary["greedy"]["ex"] == pytest.approx(0.5)
    errors = summary["greedy"]["errors"]
    assert errors["wrong_result"] == 1 and errors["schema_error"] == 1
    sampled = summary["sampled"]
    assert sampled["ex_self_consistency"] == 1.0
    assert set(sampled["maj_at_k"]) == {"1", "2", "4", "5"}
    assert sampled["pass_at_k"]["5"] == 1.0


def test_report_compares_small_and_large(tmp_path, bird_root):
    small_dir, _ = _run(tmp_path, bird_root, "small", "fine-tuned", flaky, samples=5, gpu_hourly_usd=2.0)
    large_dir, _ = _run(
        tmp_path, bird_root, "large", "frontier", oracle, input_per_mtok=2.0, output_per_mtok=10.0
    )
    out = build_report([small_dir, large_dir], tmp_path / "report", n_boot=200)
    text = out.read_text()
    assert "| small | large | greedy |" in text
    assert "| small | large | self-consistency |" in text
    payload = json.loads((tmp_path / "report" / "results.json").read_text())
    greedy_pair = next(p for p in payload["pairs"] if p["decoding"] == "greedy")
    assert greedy_pair["only_large"] == 2 and greedy_pair["only_small"] == 0
    figures = sorted(p.name for p in (tmp_path / "report" / "figures").glob("*.png"))
    assert "ex_by_model.png" in figures and "ex_by_model-dark.png" in figures
    assert "accuracy_vs_cost.png" in figures and "self_consistency.png" in figures


def test_report_and_chart_name_the_benchmark(tmp_path, bird_root):
    examples = load_examples(bird_spec(bird_root, "dev"))
    config = _config("transfer", "fine-tuned").model_copy(update={"benchmark": "spider-dev"})
    run_dir = generate(config, tmp_path / "runs", backend=FakeBackend(oracle), examples=examples)
    score_run(run_dir, examples=examples, workers=2)
    write_metrics(run_dir, n_boot=50)

    fig = plots.plot_ex(load_runs([run_dir], n_boot=50), plots.LIGHT)
    assert fig.axes[0].get_xlabel().startswith("Execution accuracy on Spider dev (%)")
    plots.plt.close(fig)
    text = build_report([run_dir], tmp_path / "report", n_boot=50, plots=False).read_text()
    assert "Benchmark: Spider dev (4 questions)." in text


def test_query_cost_accounts_for_cache():
    pricing = PricingSpec(
        input_per_mtok=4, output_per_mtok=20, cache_read_per_mtok=0.4, cache_write_per_mtok=5
    )
    usage = {
        "input_tokens": 1_000_000,
        "cached_input_tokens": 600_000,
        "cache_write_tokens": 200_000,
        "output_tokens": 100_000,
    }
    # 200k fresh * 4 + 600k * 0.4 + 200k * 5 + 100k * 20, all per million
    assert query_cost_usd(usage, pricing) == pytest.approx(0.8 + 0.24 + 1.0 + 2.0)
    gpu = PricingSpec(gpu_hourly_usd=3.6, num_gpus=2)
    assert query_cost_usd({"latency_s": 1.0}, gpu) == pytest.approx(0.002)
    assert query_cost_usd({"latency_s": 1.0}, PricingSpec()) is None
