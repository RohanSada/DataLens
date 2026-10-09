"""Training run configuration (one YAML file per run).

The ``trainer`` section is passed straight to TRL's ``SFTConfig`` or
``GRPOConfig``, so every TRL hyperparameter is available without being
re-declared here; unknown keys fail loudly when the config is built.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from datalens.config_utils import apply_overrides, load_yaml
from datalens.eval.config import PromptSpec


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class DataSpec(_Strict):
    data_root: str = "data/bird"
    benchmark: str = "bird-train"
    train_fraction: float = Field(1.0, gt=0, le=1)
    seed: int = 0
    limit: int | None = None
    max_prompt_tokens: int = Field(
        4096, description="Drop examples whose prompt is longer (the schema would be cut)."
    )
    drop_failed_gold: bool = Field(True, description="Drop examples whose gold SQL errors or times out.")
    drop_empty_gold: bool = Field(
        True, description="Drop examples whose gold SQL returns no rows (any empty answer would score)."
    )
    exclude_ids_file: str | None = Field(
        None, description="Text file of example ids to skip, e.g. ones the base model always solves."
    )
    gold_timeout_s: float = 60.0
    schema_cache_dir: str | None = ".cache/schemas"


class LoraSpec(_Strict):
    r: int = 32
    alpha: int = 64
    dropout: float = 0.05
    target_modules: str | list[str] = "all-linear"


class RewardSpec(_Strict):
    correct: float = 1.0
    executable: float = 0.1
    invalid: float = 0.0
    timeout_s: float = 10.0
    num_workers: int = 16
    max_rows: int = Field(100_000, description="Rows fetched per rollout before it counts as runaway.")
    format_weight: float = Field(0.0, description="Weight of the optional format reward.")


class ExportSpec(_Strict):
    merge_lora: bool = Field(True, description="Save a merged full model next to the adapter.")
    hub_repo: str | None = Field(None, description="Push the merged model here, e.g. user/name.")


class TrainConfig(_Strict):
    run_name: str
    method: Literal["sft", "grpo"]
    model: str = Field(description="Base model: Hugging Face id or local path.")
    output_dir: str
    data: DataSpec = Field(default_factory=DataSpec)
    prompt: PromptSpec = Field(default_factory=PromptSpec)
    lora: LoraSpec | None = Field(default_factory=LoraSpec, description="null for full fine-tuning.")
    reward: RewardSpec = Field(default_factory=RewardSpec)
    export: ExportSpec = Field(default_factory=ExportSpec)
    trainer: dict[str, Any] = Field(default_factory=dict)
    resume: bool = Field(
        True, description="Continue from the newest checkpoint in output_dir, e.g. after a disconnect."
    )

    @classmethod
    def from_yaml(cls, path: str | Path, sets: Sequence[str] = ()) -> TrainConfig:
        """Load a config; ``sets`` are ``key.path=value`` overrides."""
        return cls.model_validate(apply_overrides(load_yaml(path), sets))

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)
