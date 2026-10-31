"""Settings for an agentic evaluation run (the ``agent:`` section of an eval YAML)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class AgentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Candidate generation
    rollouts: int = Field(8, ge=1, description="Independent agent attempts per question.")
    greedy_first: bool = Field(
        True, description="Decode the first attempt greedily (the 'agent x1' ablation)."
    )
    temperature: float | None = Field(0.8, description="Sampling temperature for the other attempts.")
    top_p: float | None = 0.95
    seed: int = 0
    tools: bool = Field(True, description="False: one-shot answer, no database access (an ablation).")
    max_turns: int = Field(8, ge=1, description="Model turns per attempt, including the answer.")
    max_tokens_per_turn: int = 2048
    stop_sequences: bool = Field(
        True, description="Stop at closing tags. Turn off for models that reason in <think> blocks."
    )
    retry_empty: bool = Field(True, description="Ask once more when the answer returns no rows.")
    max_context_chars: int = Field(
        90_000, description="Attempts whose conversation grows past this are made to answer."
    )

    # What the agent sees
    descriptions: bool = Field(True, description="BIRD column and value descriptions in the schema.")
    num_examples: int = 3
    value_hints: bool = Field(True, description="List question phrases found verbatim in the data.")
    max_schema_chars: int = Field(
        60_000, description="Longer schemas drop descriptions, then examples, to fit the context."
    )
    fewshot: int = Field(0, ge=0, description="Similar solved train questions shown as examples.")
    fewshot_benchmark: str = Field("bird-train", description="Where examples come from (never the same db).")
    fewshot_data_root: str | None = Field(None, description="Defaults to the run's data_root.")
    preview_rows: int = 10
    exec_timeout_s: float = 10.0

    # Choosing among the candidates
    selection: Literal["vote", "selector"] = "selector"
    consensus: float = Field(
        0.75, description="If the largest result group has at least this share of attempts, it wins outright."
    )
    selector_candidates: int = Field(4, ge=2, description="Distinct results shown to the selector.")
    selector_samples: int = Field(3, ge=1, description="Selector judgements per question (they vote).")
    selector_temperature: float | None = 0.7
    selector_max_tokens: int = 2048
    vote_weight: float = Field(
        0.5, description="Selector score = selector votes + vote_weight * share of attempts in the group."
    )

    # Batching
    chunk_size: int = Field(64, ge=1, description="Questions whose attempts run in lockstep.")
    chat_kwargs: dict[str, Any] = Field(
        default_factory=dict, description="Extra arguments for the chat backend, merged over backend_kwargs."
    )
