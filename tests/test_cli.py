"""The CLI end to end on a fake model, and the config override syntax it exposes."""

from __future__ import annotations

import json

import pytest
import yaml
from typer.testing import CliRunner

from datalens import cli
from datalens.config_utils import apply_overrides, load_yaml
from datalens.inference.backends import FakeBackend
from datalens.prompts import PromptParts

runner = CliRunner()


def test_overrides_parse_yaml_values_and_create_parents():
    raw = {"trainer": {"learning_rate": 1.0e-5}, "data": None}
    apply_overrides(
        raw,
        [
            "trainer.learning_rate=2e-5",
            "trainer.reward_weights=[1.0, 0.5]",
            "data.train_fraction=0.25",
            "prompt.reasoning=false",
            "export.hub_repo=null",
            "model.name=Qwen2.5-Coder-1.5B SFT (0.1)",
        ],
    )
    assert raw["trainer"] == {"learning_rate": 2e-5, "reward_weights": [1.0, 0.5]}
    assert raw["data"] == {"train_fraction": 0.25}
    assert raw["prompt"]["reasoning"] is False
    assert raw["export"]["hub_repo"] is None
    assert raw["model"]["name"] == "Qwen2.5-Coder-1.5B SFT (0.1)"


@pytest.mark.parametrize("bad", ["no-equals-sign", "=1", "trainer.learning_rate.x=1"])
def test_overrides_reject_malformed_assignments(bad):
    with pytest.raises(ValueError):
        apply_overrides({"trainer": {"learning_rate": 1e-5}}, [bad])


def test_yaml_reads_exponent_floats(tmp_path):
    # Plain PyYAML reads 2e-5 as a string, which would reach the optimizer as one.
    path = tmp_path / "config.yaml"
    path.write_text("a: 2e-5\nb: 1.0e-5\nc: 3E2\nd: 1e3x\ne: '2e-5'\n", encoding="utf-8")
    assert load_yaml(path) == {"a": 2e-5, "b": 1e-5, "c": 300.0, "d": "1e3x", "e": "2e-5"}


def test_train_command_checks_the_config_method(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # a short path keeps the error on one line
    (tmp_path / "grpo.yaml").write_text(
        "run_name: x\nmethod: grpo\nmodel: tiny\noutput_dir: out\n", encoding="utf-8"
    )
    result = runner.invoke(cli.app, ["train", "sft", "grpo.yaml"])
    assert result.exit_code == 2
    assert "grpo.yaml is a grpo config, not sft" in result.output


def _count_customers(prompt: PromptParts, i: int) -> str:
    """Right only on "How many customers"; the second sample of every question has no SQL."""
    if i == 1:
        return "I am not sure."
    return "<think>count the rows</think>\n```sql\nSELECT COUNT(*) FROM customers\n```"


def test_eval_run_overrides_then_exclude_and_report(tmp_path, bird_root, monkeypatch):
    backends = []

    def fake_create_backend(kind, model, **kwargs):
        backends.append((kind, model))
        return FakeBackend(_count_customers, latency_s=0.01)

    monkeypatch.setattr("datalens.eval.generate.create_backend", fake_create_backend)
    config = tmp_path / "tiny.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "run_name": "tiny",
                "model": {"name": "Tiny", "backend": "openai", "model": "fake", "group": "fine-tuned"},
                "data_root": str(bird_root),
                "schema_cache_dir": None,
                "pricing": {"gpu_hourly_usd": 1.0},
            }
        ),
        encoding="utf-8",
    )
    runs = tmp_path / "runs"
    args = ["eval", "run", str(config), "--runs", str(runs), "--run-name", "tiny-k4"]
    args += ["--set", "decoding.samples=4", "--set", "model.name=Tiny k4"]
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert "Tiny k4: EX 25.0%" in result.output
    assert "self-consistency@4: 25.0%" in result.output
    assert backends == [("openai", "fake")]

    run_dir = runs / "tiny-k4"
    rows = [json.loads(line) for line in (run_dir / "generations.jsonl").read_text().splitlines()]
    assert len(rows) == 4
    assert all(len(row["samples"]["completions"]) == 4 for row in rows)

    # Questions every sample gets wrong carry no GRPO signal and are listed for exclusion.
    ids_file = tmp_path / "exclude.txt"
    result = runner.invoke(
        cli.app, ["data", "exclude-solved", str(run_dir / "scored.jsonl"), "--out", str(ids_file)]
    )
    assert result.exit_code == 0, result.output
    assert ids_file.read_text().split() == ["bird-dev-1", "bird-dev-2", "bird-dev-3"]

    report_dir = tmp_path / "report"
    result = runner.invoke(cli.app, ["eval", "report", str(run_dir), "--out", str(report_dir), "--no-plots"])
    assert result.exit_code == 0, result.output
    assert "Tiny k4" in (report_dir / "results.md").read_text()
