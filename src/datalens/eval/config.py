"""Evaluation run configuration (one YAML file per model under test)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from datalens.config_utils import apply_overrides, load_yaml
from datalens.prompts import PromptConfig


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class ModelSpec(_Strict):
    """The model under test and how to reach it."""

    name: str = Field(description="Display name used in tables and plots.")
    backend: Literal["vllm", "openai", "anthropic"]
    model: str = Field(description="Hugging Face id, local path, or API model id.")
    params_b: float | None = Field(None, description="Parameter count in billions.")
    group: Literal["base", "fine-tuned", "frontier"] = Field(
        description="'fine-tuned' runs are compared against 'frontier' runs in reports."
    )
    training: str = Field("none", description="e.g. none, sft, grpo, sft+grpo.")
    train_fraction: float | None = Field(
        None, description="Share of BIRD train used; set for learning-curve runs."
    )
    backend_kwargs: dict[str, Any] = Field(default_factory=dict)


class PromptSpec(_Strict):
    reasoning: bool = True
    num_examples: int = 3
    descriptions: bool = False


class DecodingSpec(_Strict):
    greedy: bool = True
    greedy_temperature: float | None = Field(
        0.0, description="null for APIs that reject sampling parameters."
    )
    samples: int = Field(0, description="Sampled completions per question for maj@k/pass@k.")
    temperature: float | None = 0.8
    top_p: float | None = None
    max_tokens: int = 2048
    seed: int | None = 0


class PricingSpec(_Strict):
    """Unit prices. API models use token prices; local models use GPU-hours."""

    input_per_mtok: float | None = None
    output_per_mtok: float | None = None
    cache_read_per_mtok: float | None = None
    cache_write_per_mtok: float | None = None
    gpu_hourly_usd: float | None = None
    num_gpus: int = 1


class EvalConfig(_Strict):
    run_name: str
    model: ModelSpec
    benchmark: str = "bird-dev"
    data_root: str = "data/bird"
    limit: int | None = None
    batch_size: int = 64
    prompt: PromptSpec = Field(default_factory=PromptSpec)
    decoding: DecodingSpec = Field(default_factory=DecodingSpec)
    pricing: PricingSpec = Field(default_factory=PricingSpec)
    schema_cache_dir: str | None = ".cache/schemas"

    @classmethod
    def from_yaml(cls, path: str | Path, sets: Sequence[str] = (), **overrides: Any) -> EvalConfig:
        """Load a config; ``sets`` are ``key.path=value`` strings, ``overrides`` top-level fields."""
        raw = load_yaml(path)
        raw.update({k: v for k, v in overrides.items() if v is not None})
        return cls.model_validate(apply_overrides(raw, sets))

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)

    def prompt_config(self) -> PromptConfig:
        return PromptConfig(
            reasoning=self.prompt.reasoning,
            num_examples=self.prompt.num_examples,
            descriptions=self.prompt.descriptions,
            schema_cache_dir=self.schema_cache_dir,
        )
