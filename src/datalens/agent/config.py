"""Agent run configuration (one YAML file per pipeline variant).

``models`` lists the model servers (loaded lazily); ``roles`` says which model
plays each part of the paper's pipeline, optionally with a LoRA adapter, so the
reasoning generator and the selector can be adapters on one base model that a
single vLLM engine serves.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from datalens.agent.prompts import ICLStyle
from datalens.config_utils import apply_overrides, load_yaml
from datalens.eval.config import EvalConfig, PricingSpec

ROLE_NAMES = ("understanding", "reasoning_generator", "icl_generator", "fixer", "revisor", "selector")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class ModelEntry(_Strict):
    backend: Literal["vllm", "openai", "anthropic"]
    model: str = Field(description="Hugging Face id, local path, or API model id.")
    backend_kwargs: dict[str, Any] = Field(default_factory=dict)


class RoleSpec(_Strict):
    model: str = Field(description="A key of `models`.")
    lora: str | None = Field(None, description="LoRA adapter directory (vLLM models only).")


class Roles(_Strict):
    understanding: RoleSpec
    reasoning_generator: RoleSpec
    icl_generator: RoleSpec
    fixer: RoleSpec
    revisor: RoleSpec
    selector: RoleSpec

    def get(self, name: str) -> RoleSpec:
        spec: RoleSpec = getattr(self, name)
        return spec


class UnderstandingSpec(_Strict):
    enabled: bool = Field(True, description="false = the 'w/o task understanding' ablation.")
    llm: bool = Field(True, description="Extract literals and skeleton with the understanding model.")
    value_retrieval: bool = True
    match_threshold: float = 0.85
    max_matched_values: int = 20
    literal_threshold: float = 0.8
    per_literal: int = 3
    examples: int = Field(3, description="Few-shot examples retrieved for the ICL generator.")
    example_split: str = "bird-train"
    example_skeletons: Literal["llm", "heuristic"] = Field(
        "llm", description="How skeletons of the example pool are made (llm = the paper's way)."
    )
    embedding_model: str | None = Field(
        "sentence-transformers/all-MiniLM-L6-v2", description="null = TF-IDF retrieval."
    )
    temperature: float = 0.2
    max_tokens: int = 512


class ReasoningSpec(_Strict):
    enabled: bool = Field(True, description="false = the 'w/o reasoning generator' ablation.")
    n: int = 8
    temperature: float = 0.8
    top_p: float | None = None
    max_tokens: int = 2048
    matched_values: bool = Field(True, description="List matched values first among column examples.")
    num_examples: int = Field(3, description="Example values per column (the GRPO prompt uses 3).")


class ICLVariant(_Strict):
    style: ICLStyle
    temperature: float
    n: int = 1


def _default_variants() -> list[ICLVariant]:
    return [
        ICLVariant(style=s, temperature=t) for s in ("direct", "cot", "decompose") for t in (0.2, 0.6, 1.0)
    ]


class ICLSpec(_Strict):
    enabled: bool = Field(True, description="false = the 'w/o ICL generator' ablation.")
    variants: list[ICLVariant] = Field(default_factory=_default_variants)
    max_tokens: int = 2048
    shuffle_examples: bool = Field(True, description="Shuffle example order per variant, as the paper does.")


class RefineSpec(_Strict):
    fixer: bool = True
    revisor: bool = True
    max_rounds: int = Field(2, description="Fix, execute, and fix again while still failing.")
    fixer_temperature: float = 0.2
    revisor_temperature: float = 0.7
    max_tokens: int = 1536


class SelectionSpec(_Strict):
    method: Literal["tournament", "vote"] = "tournament"
    max_candidates: int | None = Field(
        16, description="Most-voted result groups that enter the tournament (null = all)."
    )
    both_orders: bool = Field(False, description="Judge each pair twice, swapping positions.")
    temperature: float = 0.0
    max_tokens: int = 1024


class ExecutionSpec(_Strict):
    timeout_s: float = 30.0
    preview_rows: int = 10
    workers: int = 8


class AgentConfig(_Strict):
    run_name: str
    name: str = Field(description="Display name in reports.")
    params_b: float | None = None
    training: str = "grpo+agentar"
    benchmark: str = "bird-dev"
    data_root: str = "data/bird"
    limit: int | None = Field(None, description="Only the first N questions.")
    sample: int | None = Field(None, description="A random subset of N questions (seeded).")
    seed: int = 0
    batch_size: int = Field(128, description="Questions per chunk; results are saved after each chunk.")
    max_model_len: int | None = Field(
        None, description="Skip prompts longer than this minus max_tokens (vLLM rejects them)."
    )
    release_models: bool = Field(True, description="Free a model's GPU memory once no later stage needs it.")
    models: dict[str, ModelEntry]
    roles: Roles
    understanding: UnderstandingSpec = Field(default_factory=UnderstandingSpec)
    reasoning: ReasoningSpec = Field(default_factory=ReasoningSpec)
    icl: ICLSpec = Field(default_factory=ICLSpec)
    refinement: RefineSpec = Field(default_factory=RefineSpec)
    selection: SelectionSpec = Field(default_factory=SelectionSpec)
    execution: ExecutionSpec = Field(default_factory=ExecutionSpec)
    pricing: PricingSpec = Field(default_factory=PricingSpec)
    schema_cache_dir: str | None = ".cache/schemas"
    index_dir: str = Field(".cache/agent", description="Value indexes and the example store.")

    @model_validator(mode="after")
    def _check_roles(self) -> AgentConfig:
        for role in ROLE_NAMES:
            spec = self.roles.get(role)
            if spec.model not in self.models:
                raise ValueError(f"role {role} uses model {spec.model!r}, which is not under `models`")
            if spec.lora and self.models[spec.model].backend != "vllm":
                raise ValueError(f"role {role}: LoRA adapters need a vllm model")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path, sets: Sequence[str] = (), **overrides: Any) -> AgentConfig:
        raw = load_yaml(path)
        raw.update({k: v for k, v in overrides.items() if v is not None})
        return cls.model_validate(apply_overrides(raw, sets))

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)

    def eval_config(self, pool_size: int) -> EvalConfig:
        """The config scoring and reports read: the agent's final answer is its 'greedy' answer."""
        generator = self.models[self.roles.reasoning_generator.model]
        model_id = self.roles.reasoning_generator.lora or generator.model
        return EvalConfig.model_validate(
            {
                "run_name": self.run_name,
                "model": {
                    "name": self.name,
                    "backend": generator.backend,
                    "model": model_id,
                    "params_b": self.params_b,
                    "group": "fine-tuned",
                    "training": self.training,
                },
                "benchmark": self.benchmark,
                "data_root": self.data_root,
                "limit": self.limit,
                "prompt": {"reasoning": True, "num_examples": self.reasoning.num_examples},
                "decoding": {"greedy": True, "samples": pool_size, "temperature": self.reasoning.temperature},
                "pricing": self.pricing.model_dump(),
                "schema_cache_dir": self.schema_cache_dir,
            }
        )
