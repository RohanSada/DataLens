"""End-to-end SFT and GRPO on a tiny randomly initialised model, on CPU.

These exercise the real TRL code paths (dataset building, chat templating, LoRA,
the execution reward inside GRPO, merging) in seconds, so a broken training run
is caught here instead of an hour into a GPU job. Model quality is irrelevant.
"""

from __future__ import annotations

import json
import logging

import pytest

pytest.importorskip("trl")
pytest.importorskip("peft")

from datalens.training.config import TrainConfig
from datalens.training.train import run

pytestmark = pytest.mark.smoke

CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def tiny_model_and_tokenizer(corpus: list[str]):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    specials = ["<unk>", "<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(
        corpus,
        trainers.BpeTrainer(
            vocab_size=600, special_tokens=specials, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        ),
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="<unk>", eos_token="<|im_end|>", pad_token="<|endoftext|>"
    )
    tokenizer.chat_template = CHAT_TEMPLATE

    config = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        tie_word_embeddings=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=None,
    )
    model = Qwen2ForCausalLM(config)
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    return model, tokenizer


def _corpus(bird_root) -> list[str]:
    from datalens.data.benchmarks import bird_spec, load_examples
    from datalens.prompts import PromptConfig, format_answer, prompt_for_example

    texts = []
    for example in load_examples(bird_spec(bird_root, "dev")):
        texts.append(prompt_for_example(example, PromptConfig(schema_cache_dir=None)).user)
        texts.append(format_answer(example.gold_sql, reasoning="think"))
    return texts * 5


def _config(tmp_path, bird_root, method: str, trainer: dict) -> TrainConfig:
    return TrainConfig.model_validate(
        {
            "run_name": f"smoke-{method}",
            "method": method,
            "model": "tiny",
            "output_dir": str(tmp_path / f"out-{method}"),
            "data": {"data_root": str(bird_root), "benchmark": "bird-dev", "schema_cache_dir": None},
            "prompt": {"reasoning": method == "grpo"},
            "lora": {"r": 4, "alpha": 8, "dropout": 0.0},
            "trainer": {
                "max_steps": 2,
                "learning_rate": 1e-3,
                "logging_steps": 1,
                "save_strategy": "no",
                "report_to": "none",
                "use_cpu": True,
                "bf16": False,
                "gradient_checkpointing": False,
                **trainer,
            },
        }
    )


def test_sft_smoke(tmp_path, bird_root, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model, tokenizer = tiny_model_and_tokenizer(_corpus(bird_root))
    config = _config(tmp_path, bird_root, "sft", {"per_device_train_batch_size": 2, "max_length": 2048})
    out = run(config, model=model, tokenizer=tokenizer)
    assert (out / "adapter_config.json").is_file()
    assert (out / "merged" / "config.json").is_file()
    assert len(json.loads((out / "train_ids.json").read_text())) == 4


def test_grpo_smoke(tmp_path, bird_root, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model, tokenizer = tiny_model_and_tokenizer(_corpus(bird_root))
    config = _config(
        tmp_path,
        bird_root,
        "grpo",
        {
            "per_device_train_batch_size": 4,
            "num_generations": 2,
            "max_completion_length": 12,
            "temperature": 1.0,
        },
    )
    out = run(config, model=model, tokenizer=tokenizer)
    state = json.loads((out / "trainer_state.json").read_text())
    logged = [h for h in state["log_history"] if "sql/exec_accuracy" in h]
    assert logged, "the execution reward should log its metrics during training"
    assert all(0.0 <= h["sql/executable"] <= 1.0 for h in logged)
    assert (out / "merged" / "config.json").is_file()


def test_grpo_updates_weights_when_rewards_vary(tmp_path, bird_root, monkeypatch):
    """A random model never writes valid SQL, so all real rewards tie and the
    advantage is zero. Give rollouts varying rewards to exercise the update path."""
    from datalens.rewards import ExecutionReward

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ExecutionReward, "score", lambda self, text, db, gold: float(len(text) % 2))
    model, tokenizer = tiny_model_and_tokenizer(_corpus(bird_root))
    config = _config(
        tmp_path,
        bird_root,
        "grpo",
        {"per_device_train_batch_size": 8, "num_generations": 4, "max_completion_length": 24},
    )
    config.export.merge_lora = False
    out = run(config, model=model, tokenizer=tokenizer)
    state = json.loads((out / "trainer_state.json").read_text())
    assert any(h.get("grad_norm", 0) > 0 for h in state["log_history"])


def test_grpo_rerun_resumes_from_the_last_checkpoint(tmp_path, bird_root, monkeypatch, caplog):
    """Rerunning after a disconnect continues from the newest checkpoint, not step 0."""
    from datalens.rewards import ExecutionReward

    monkeypatch.chdir(tmp_path)
    scored: list[int] = []
    score = ExecutionReward.score
    monkeypatch.setattr(ExecutionReward, "score", lambda self, *args: scored.append(1) or score(self, *args))
    trainer = {
        "per_device_train_batch_size": 4,
        "num_generations": 2,
        "max_completion_length": 12,
        "save_strategy": "steps",
        "save_steps": 1,
    }
    config = _config(tmp_path, bird_root, "grpo", trainer)
    config.export.merge_lora = False
    model, tokenizer = tiny_model_and_tokenizer(_corpus(bird_root))
    out = run(config, model=model, tokenizer=tokenizer)
    assert (out / "checkpoint-2").is_dir()
    first_run = len(scored)

    scored.clear()
    config.trainer["max_steps"] = 3
    model, tokenizer = tiny_model_and_tokenizer(_corpus(bird_root))
    with caplog.at_level(logging.INFO, logger="datalens.training.train"):
        run(config, model=model, tokenizer=tokenizer)
    assert f"resuming from {out / 'checkpoint-2'}" in caplog.text
    assert json.loads((out / "trainer_state.json").read_text())["global_step"] == 3
    assert 2 * len(scored) == first_run, "only the one new step should run"
