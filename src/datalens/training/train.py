"""SFT and GRPO training entry points (TRL + PEFT).

``run(config)`` is the whole pipeline: filter BIRD train, build the dataset,
train, save the adapter, then optionally merge it into a standalone model (what
vLLM serves) and push that to the Hugging Face Hub. Running the same config again
continues from the newest checkpoint in ``output_dir``, so a long job that was
interrupted picks up where it stopped.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from datalens.rewards import ExecutionReward, RewardConfig, format_reward
from datalens.training.config import TrainConfig
from datalens.training.data import Messages, build_records, select_examples

log = logging.getLogger(__name__)


def _tokenizer(model: str) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _token_counter(tokenizer: Any) -> Any:
    def count(messages: Messages) -> int:
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        # transformers may return a BatchEncoding or a plain list of ids
        if hasattr(ids, "keys") and "input_ids" in ids:
            ids = ids["input_ids"]
        return len(ids)

    return count


def _peft_config(config: TrainConfig) -> Any:
    if config.lora is None:
        return None
    from peft import LoraConfig

    return LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        task_type="CAUSAL_LM",
    )


def build_dataset(config: TrainConfig, tokenizer: Any) -> Any:
    from datasets import Dataset

    examples = select_examples(config.data)
    records = build_records(
        examples,
        method=config.method,
        prompt=config.prompt,
        schema_cache_dir=config.data.schema_cache_dir,
        count_tokens=_token_counter(tokenizer),
        max_prompt_tokens=config.data.max_prompt_tokens,
    )
    if not records:
        raise ValueError("no training examples left after filtering")
    return Dataset.from_list(records)


def make_trainer(config: TrainConfig, dataset: Any, tokenizer: Any, model: Any = None) -> Any:
    """Construct the TRL trainer. ``model`` overrides ``config.model`` (tests pass a tiny model)."""
    model = model if model is not None else config.model
    trainer_args: dict[str, Any] = {"output_dir": config.output_dir, "run_name": config.run_name}
    trainer_args.update(config.trainer)

    if config.method == "sft":
        from trl import SFTConfig, SFTTrainer

        return SFTTrainer(
            model=model,
            args=SFTConfig(**trainer_args),
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=_peft_config(config),
        )

    from trl import GRPOConfig, GRPOTrainer

    reward = ExecutionReward(
        RewardConfig(
            correct=config.reward.correct,
            executable=config.reward.executable,
            invalid=config.reward.invalid,
            timeout_s=config.reward.timeout_s,
            num_workers=config.reward.num_workers,
            max_rows=config.reward.max_rows,
        )
    )
    reward_funcs: list[Any] = [reward]
    if config.reward.format_weight > 0:
        reward_funcs.append(format_reward)
        trainer_args.setdefault("reward_weights", [1.0, config.reward.format_weight])
    return GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=GRPOConfig(**trainer_args),
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=_peft_config(config),
    )


def export_model(config: TrainConfig, trainer: Any, tokenizer: Any) -> Path | None:
    """Merge LoRA weights into the base model so vLLM can serve one directory."""
    if config.lora is None or not config.export.merge_lora:
        return None
    merged_dir = Path(config.output_dir) / "merged"
    model = trainer.model
    if hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()
    model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    log.info("merged model saved to %s", merged_dir)
    if config.export.hub_repo:
        model.push_to_hub(config.export.hub_repo)
        tokenizer.push_to_hub(config.export.hub_repo)
        log.info("pushed merged model to %s", config.export.hub_repo)
    return merged_dir


def _last_checkpoint(out: Path) -> str | None:
    from transformers.trainer_utils import get_last_checkpoint

    return get_last_checkpoint(str(out))


def run(config: TrainConfig, *, model: Any = None, tokenizer: Any = None, dataset: Any = None) -> Path:
    """Train according to ``config``; returns the output directory."""
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "datalens_train_config.yaml").write_text(config.to_yaml(), encoding="utf-8")

    tokenizer = tokenizer if tokenizer is not None else _tokenizer(config.model)
    dataset = dataset if dataset is not None else build_dataset(config, tokenizer)
    log.info("%s on %d examples -> %s", config.method.upper(), len(dataset), out)
    (out / "train_ids.json").write_text(json.dumps(list(dataset["id"])), encoding="utf-8")

    if config.method == "sft":
        dataset = dataset.remove_columns(["id"])  # SFT collators expect model inputs only
    trainer = make_trainer(config, dataset, tokenizer, model=model)
    checkpoint = _last_checkpoint(out) if config.resume else None
    if checkpoint:
        log.info("resuming from %s", checkpoint)
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(str(out))
    trainer.save_state()
    export_model(config, trainer, tokenizer)
    return out
