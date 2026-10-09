"""Model backends behind one interface.

Every backend takes a batch of prompts and returns ``n`` completions per prompt
with token counts and timing, so the evaluation harness can compare a 1.5B model
on a local GPU with a frontier model behind an API on accuracy *and* cost.

* ``vllm``: offline batched generation for local or Hugging Face models.
* ``openai``: any OpenAI-compatible server (``vllm serve``, OpenAI, etc.).
* ``anthropic``: Claude models through the Anthropic SDK, with prompt caching.
* ``fake``: deterministic scripted outputs for tests and demos.

Heavy SDKs are imported lazily so the core package installs without them.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Protocol

from datalens.prompts import PromptParts


@dataclass
class Completion:
    text: str
    output_tokens: int | None = None
    finish_reason: str | None = None


@dataclass
class Generation:
    """All completions for one prompt, plus what they cost to produce."""

    completions: list[Completion]
    input_tokens: int | None = None  # all prompt tokens, cached or not
    cached_input_tokens: int = 0  # prompt tokens read from a prompt cache
    cache_write_tokens: int = 0  # prompt tokens written to a prompt cache
    latency_s: float = 0.0  # wall time attributable to this prompt

    @property
    def output_tokens(self) -> int | None:
        counts = [c.output_tokens for c in self.completions]
        if any(c is None for c in counts):
            return None
        return sum(c for c in counts if c is not None)


@dataclass(frozen=True)
class SamplingConfig:
    n: int = 1
    temperature: float | None = 0.0
    top_p: float | None = None
    max_tokens: int = 2048
    seed: int | None = None


class Backend(Protocol):
    name: str

    def generate(self, prompts: Sequence[PromptParts], sampling: SamplingConfig) -> list[Generation]:
        """Return one :class:`Generation` per prompt, in order."""
        ...


@dataclass
class FakeBackend:
    """Returns ``responder(prompt, sample_index)`` for every completion."""

    responder: Callable[[PromptParts, int], str]
    name: str = "fake"
    latency_s: float = 0.0
    calls: list[int] = field(default_factory=list)

    def generate(self, prompts: Sequence[PromptParts], sampling: SamplingConfig) -> list[Generation]:
        self.calls.append(len(prompts))
        out = []
        for prompt in prompts:
            completions = [
                Completion(text=self.responder(prompt, i), output_tokens=10, finish_reason="stop")
                for i in range(sampling.n)
            ]
            out.append(
                Generation(
                    completions=completions, input_tokens=len(prompt.user) // 4, latency_s=self.latency_s
                )
            )
        return out


class VLLMBackend:
    """Offline batched generation with ``vllm.LLM``.

    Latency is the batch wall time divided by batch size. That is the amortised
    GPU time per question, which is what cost-per-query is computed from.
    """

    def __init__(self, model: str, **engine_kwargs: Any) -> None:
        from vllm import LLM  # lazy: only installed with the [vllm] extra

        engine_kwargs.setdefault("enable_prefix_caching", True)
        self.name = model
        self.llm = LLM(model=model, **engine_kwargs)

    def generate(self, prompts: Sequence[PromptParts], sampling: SamplingConfig) -> list[Generation]:
        from vllm import SamplingParams

        greedy = not sampling.temperature
        params = SamplingParams(
            n=1 if greedy else sampling.n,
            temperature=0.0 if greedy else sampling.temperature,
            top_p=sampling.top_p or 1.0,
            max_tokens=sampling.max_tokens,
            seed=sampling.seed,
        )
        start = time.perf_counter()
        outputs = self.llm.chat([p.messages() for p in prompts], params, use_tqdm=False)
        per_prompt = (time.perf_counter() - start) / max(len(prompts), 1)
        return [
            Generation(
                completions=[
                    Completion(
                        text=o.text,
                        output_tokens=len(o.token_ids),
                        finish_reason=o.finish_reason,
                    )
                    for o in out.outputs
                ],
                input_tokens=len(out.prompt_token_ids or []),
                latency_s=per_prompt,
            )
            for out in outputs
        ]


class OpenAICompatBackend:
    """Chat completions against any OpenAI-compatible endpoint."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        concurrency: int = 16,
        supports_n: bool = True,
        timeout_s: float = 300.0,
    ) -> None:
        from openai import OpenAI

        self.name = model
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY", timeout=timeout_s)
        self.concurrency = concurrency
        self.supports_n = supports_n

    def _one(self, prompt: PromptParts, sampling: SamplingConfig) -> Generation:
        kwargs: dict[str, Any] = {"max_tokens": sampling.max_tokens}
        if sampling.temperature is not None:
            kwargs["temperature"] = sampling.temperature
        if sampling.top_p is not None:
            kwargs["top_p"] = sampling.top_p
        if sampling.seed is not None:
            kwargs["seed"] = sampling.seed
        rounds = 1 if self.supports_n else sampling.n
        if self.supports_n and sampling.n > 1:
            kwargs["n"] = sampling.n

        completions: list[Completion] = []
        input_tokens = 0
        start = time.perf_counter()
        for _ in range(rounds):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=prompt.messages(),  # type: ignore[arg-type]
                **kwargs,
            )
            usage = response.usage
            input_tokens += usage.prompt_tokens if usage else 0
            per_choice = (usage.completion_tokens // max(len(response.choices), 1)) if usage else None
            for choice in response.choices:
                completions.append(
                    Completion(
                        text=choice.message.content or "",
                        output_tokens=per_choice,
                        finish_reason=choice.finish_reason,
                    )
                )
        return Generation(
            completions=completions,
            input_tokens=input_tokens,
            latency_s=time.perf_counter() - start,
        )

    def generate(self, prompts: Sequence[PromptParts], sampling: SamplingConfig) -> list[Generation]:
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            return list(pool.map(lambda p: self._one(p, sampling), prompts))


class AnthropicBackend:
    """Claude through the Anthropic Messages API.

    The schema part of the prompt is marked for prompt caching: questions about
    the same database share it, so after the first question most input tokens
    are cache reads. Examples should be sent grouped by database (the runner
    does this) because cache entries expire after a few minutes.

    Current Opus and Sonnet models always think adaptively and reject custom
    sampling parameters, so ``temperature`` is only sent when configured (for
    older models), and ``effort`` controls how much the model reasons. Responses
    are streamed, which keeps long thinking requests from hitting idle timeouts.
    Server-side model fallbacks are deliberately *not* enabled: a fallback would
    answer with a different model and silently corrupt the comparison. Refusals
    are recorded as empty completions with ``finish_reason="refusal"``.
    """

    def __init__(
        self,
        model: str,
        *,
        effort: str | None = None,
        thinking: dict[str, Any] | None = None,
        concurrency: int = 8,
        cache_schema: bool = True,
        timeout_s: float = 600.0,
        max_retries: int = 4,
        client: Any = None,
    ) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic(timeout=timeout_s, max_retries=max_retries)
        self.name = model
        self.model = model
        self.effort = effort
        self.thinking = thinking
        self.concurrency = concurrency
        self.cache_schema = cache_schema
        self.client = client

    def _request(self, prompt: PromptParts, sampling: SamplingConfig) -> dict[str, Any]:
        schema_block: dict[str, Any] = {"type": "text", "text": prompt.schema_block}
        if self.cache_schema:
            schema_block["cache_control"] = {"type": "ephemeral"}
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": sampling.max_tokens,
            "system": prompt.system,
            "messages": [
                {
                    "role": "user",
                    "content": [schema_block, {"type": "text", "text": prompt.question_block}],
                }
            ],
        }
        if sampling.temperature is not None:
            # No longer a typed SDK argument, since current models reject it.
            request["extra_body"] = {"temperature": sampling.temperature}
        if self.effort:
            request["output_config"] = {"effort": self.effort}
        if self.thinking:
            request["thinking"] = self.thinking
        return request

    def _sample(self, request: dict[str, Any]) -> Generation:
        start = time.perf_counter()
        with self.client.messages.stream(**request) as stream:
            response = stream.get_final_message()
        usage = response.usage
        cache_read = usage.cache_read_input_tokens or 0
        cache_write = usage.cache_creation_input_tokens or 0
        text = "".join(block.text for block in response.content if block.type == "text")
        if response.stop_reason == "refusal":
            text = ""
        return Generation(
            completions=[
                Completion(text=text, output_tokens=usage.output_tokens, finish_reason=response.stop_reason)
            ],
            input_tokens=usage.input_tokens + cache_read + cache_write,
            cached_input_tokens=cache_read,
            cache_write_tokens=cache_write,
            latency_s=time.perf_counter() - start,
        )

    def generate(self, prompts: Sequence[PromptParts], sampling: SamplingConfig) -> list[Generation]:
        """One request per sample, all run concurrently.

        A prompt's latency is that of its slowest sample, i.e. the time to answer
        it when its samples run in parallel.
        """
        requests = [self._request(p, sampling) for p in prompts]
        jobs = [request for request in requests for _ in range(sampling.n)]
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            samples = list(pool.map(self._sample, jobs))
        out = []
        for i in range(len(prompts)):
            mine = samples[i * sampling.n : (i + 1) * sampling.n]
            out.append(
                Generation(
                    completions=[s.completions[0] for s in mine],
                    input_tokens=sum(s.input_tokens or 0 for s in mine),
                    cached_input_tokens=sum(s.cached_input_tokens for s in mine),
                    cache_write_tokens=sum(s.cache_write_tokens for s in mine),
                    latency_s=max((s.latency_s for s in mine), default=0.0),
                )
            )
        return out


def create_backend(kind: str, model: str, **kwargs: Any) -> Backend:
    """Instantiate a backend by name (``vllm``, ``openai``, ``anthropic``)."""
    if kind == "vllm":
        return VLLMBackend(model, **kwargs)
    if kind == "openai":
        return OpenAICompatBackend(model, **kwargs)
    if kind == "anthropic":
        return AnthropicBackend(model, **kwargs)
    raise ValueError(f"unknown backend {kind!r}; expected vllm, openai or anthropic")
