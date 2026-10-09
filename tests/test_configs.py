"""Every shipped config loads, and training and evaluation configs agree with each other."""

from __future__ import annotations

from pathlib import Path

import pytest

from datalens.eval.config import EvalConfig
from datalens.training.config import TrainConfig

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
TRAIN_CONFIGS = sorted((CONFIGS / "train").glob("*.yaml"))
EVAL_CONFIGS = sorted((CONFIGS / "eval").glob("*.yaml"))


def test_configs_are_found():
    assert len(TRAIN_CONFIGS) >= 5
    assert len(EVAL_CONFIGS) >= 10


@pytest.mark.parametrize("path", EVAL_CONFIGS, ids=lambda p: p.stem)
def test_eval_config_loads(path):
    cfg = EvalConfig.from_yaml(path)
    assert cfg.run_name == path.stem
    if cfg.model.backend == "anthropic":
        assert cfg.decoding.greedy_temperature is None  # current Claude models reject temperature
        assert cfg.pricing.input_per_mtok and cfg.pricing.output_per_mtok
    else:
        assert cfg.pricing.gpu_hourly_usd


@pytest.mark.parametrize("path", TRAIN_CONFIGS, ids=lambda p: p.stem)
def test_train_config_loads(path):
    cfg = TrainConfig.from_yaml(path)
    assert cfg.run_name == path.stem
    assert path.stem.startswith(cfg.method)
    assert isinstance(cfg.trainer["learning_rate"], float)


def test_fine_tuned_evals_match_their_training_runs():
    """A fine-tuned model is evaluated from its merged checkpoint, with the prompt it was trained on."""
    fine_tuned = [EvalConfig.from_yaml(p) for p in EVAL_CONFIGS]
    fine_tuned = [c for c in fine_tuned if c.model.group == "fine-tuned"]
    assert fine_tuned
    for cfg in fine_tuned:
        train = TrainConfig.from_yaml(CONFIGS / "train" / f"{cfg.run_name}.yaml")
        assert cfg.model.model == f"{train.output_dir}/merged"
        assert cfg.model.training == train.method
        assert cfg.prompt.reasoning == train.prompt.reasoning
        assert cfg.prompt.num_examples == train.prompt.num_examples


@pytest.mark.smoke
@pytest.mark.parametrize("path", TRAIN_CONFIGS, ids=lambda p: p.stem)
def test_trainer_section_is_a_valid_trl_config(path, tmp_path):
    pytest.importorskip("trl")
    from trl import GRPOConfig, SFTConfig

    cfg = TrainConfig.from_yaml(path)
    trl_config = SFTConfig if cfg.method == "sft" else GRPOConfig
    args = trl_config(output_dir=str(tmp_path), **{**cfg.trainer, "bf16": False})  # CI has no GPU
    assert args.learning_rate == cfg.trainer["learning_rate"]
