"""Multi-turn chat backends for the agent.

The agent runs many conversations in lockstep: every active rollout of every
question in a chunk takes its next turn in one batched call. With vLLM that is
one ``LLM.chat`` call, so the GPU sees hundreds of sequences at once and the
shared schema prefix is computed once per database (prefix caching). API
backends run the same batch on a thread pool.

A message is ``{"role": ..., "content": str}``. The first user message may also
carry ``"parts"``, the same text split where a prompt cache should break (schema,
then question). Backends without prompt caching ignore it.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Protocol

Message = dict[str, Any]
Conversation = list[Message]


@dataclass(frozen=True)
class ChatSampling:
    temperature: float | None = 0.0
    top_p: float | None = None
    max_tokens: int = 1536
    seed: int | None = None
    stop: tuple[str, ...] = ()


@dataclass
class ChatResult:
    text: str
    finish_reason: str | None = None
    stop_sequence: str | None = None  # which stop string ended the turn, if any
    input_tokens: int = 0  # all prompt tokens, cached or not
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0


class ChatModel(Protocol):
    name: str

    def chat(
        self, conversations: Sequence[Conversation], sampling: Sequence[ChatSampling]
    ) -> list[ChatResult]:
        """One reply per conversation, in order."""
        ...


def plain(conversation: Conversation) -> list[dict[str, str]]:
    return [{"role": m["role"], "content": m["content"]} for m in conversation]


@dataclass
class FakeChat:
    """Scripted replies for tests: ``responder(conversation, sampling) -> text``."""

    responder: Callable[[Conversation, ChatSampling], str]
    name: str = "fake"
    calls: list[int] = field(default_factory=list)

    def chat(
        self, conversations: Sequence[Conversation], sampling: Sequence[ChatSampling]
    ) -> list[ChatResult]:
        self.calls.append(len(conversations))
        out = []
        for conv, samp in zip(conversations, sampling, strict=True):
            text = self.responder(conv, samp)
            stop = None
            for s in samp.stop:
                if s in text:
                    text = text[: text.index(s)]
                    stop = s
                    break
            out.append(
                ChatResult(
                    text=text,
                    finish_reason="stop",
                    stop_sequence=stop,
                    input_tokens=sum(len(m["content"]) for m in conv) // 4,
                    output_tokens=max(1, len(text) // 4),
                )
            )
        return out


class VLLMChat:
    """Offline batched chat with ``vllm.LLM`` (prefix caching on)."""

    def __init__(
        self, model: str, *, chat_template_kwargs: dict[str, Any] | None = None, **engine_kwargs: Any
    ):
        from vllm import LLM  # lazy: only installed with the [vllm] extra

        engine_kwargs.setdefault("enable_prefix_caching", True)
        self.name = model
        self.chat_template_kwargs = chat_template_kwargs
        self.llm = LLM(model=model, **engine_kwargs)

    def chat(
        self, conversations: Sequence[Conversation], sampling: Sequence[ChatSampling]
    ) -> list[ChatResult]:
        from vllm import SamplingParams

        if not conversations:
            return []
        params = [
            SamplingParams(
                n=1,
                temperature=s.temperature or 0.0,
                top_p=s.top_p or 1.0,
                max_tokens=s.max_tokens,
                seed=s.seed,
                stop=list(s.stop) or None,
            )
            for s in sampling
        ]
        start = time.perf_counter()
        kwargs: dict[str, Any] = {"use_tqdm": False}
        if self.chat_template_kwargs:
            kwargs["chat_template_kwargs"] = self.chat_template_kwargs
        outputs = self.llm.chat([plain(c) for c in conversations], params, **kwargs)
        per = (time.perf_counter() - start) / len(conversations)
        results = []
        for out in outputs:
            o = out.outputs[0]
            stop = o.stop_reason if isinstance(o.stop_reason, str) else None
            results.append(
                ChatResult(
                    text=o.text,
                    finish_reason=o.finish_reason,
                    stop_sequence=stop,
                    input_tokens=len(out.prompt_token_ids or []),
                    cached_input_tokens=int(getattr(out, "num_cached_tokens", 0) or 0),
                    output_tokens=len(o.token_ids),
                    latency_s=per,
                )
            )
        return results


class OpenAIChat:
    """Any OpenAI-compatible chat endpoint (``vllm serve``, OpenRouter, Gemini's endpoint, ...)."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        concurrency: int = 16,
        timeout_s: float = 300.0,
        max_retries: int = 4,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        from openai import OpenAI

        self.name = model
        self.model = model
        self.client = OpenAI(
            base_url=base_url, api_key=api_key or "EMPTY", timeout=timeout_s, max_retries=max_retries
        )
        self.concurrency = concurrency
        self.extra_body = extra_body

    def _one(self, conversation: Conversation, sampling: ChatSampling) -> ChatResult:
        kwargs: dict[str, Any] = {"max_tokens": sampling.max_tokens}
        if sampling.temperature is not None:
            kwargs["temperature"] = sampling.temperature
        if sampling.top_p is not None:
            kwargs["top_p"] = sampling.top_p
        if sampling.seed is not None:
            kwargs["seed"] = sampling.seed
        if sampling.stop:
            kwargs["stop"] = list(sampling.stop)[:4]
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=plain(conversation),  # type: ignore[arg-type]
            **kwargs,
        )
        choice = response.choices[0]
        text = choice.message.content or ""
        usage = response.usage
        # The API doesn't say which stop string fired; infer it from the open tag.
        stop = None
        if choice.finish_reason == "stop" and sampling.stop:
            for s in sampling.stop:
                if text.rfind("<" + s.strip("</>")) > text.rfind(s):
                    stop = s
                    break
        return ChatResult(
            text=text,
            finish_reason=choice.finish_reason,
            stop_sequence=stop,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_s=time.perf_counter() - start,
        )

    def chat(
        self, conversations: Sequence[Conversation], sampling: Sequence[ChatSampling]
    ) -> list[ChatResult]:
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            return list(pool.map(self._one, conversations, sampling))


class AnthropicChat:
    """Claude through the Messages API, with the schema prefix prompt-cached.

    Current Claude models reject sampling parameters, so temperature is never
    sent; rollouts differ through the model's own sampling. ``effort`` sets how
    much the model thinks. Thinking blocks are not kept between turns: each turn
    is sent as plain text, which keeps conversations portable across backends.
    """

    def __init__(
        self,
        model: str,
        *,
        effort: str | None = None,
        concurrency: int = 8,
        timeout_s: float = 600.0,
        max_retries: int = 6,
        client: Any = None,
    ) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic(timeout=timeout_s, max_retries=max_retries)
        self.name = model
        self.model = model
        self.effort = effort
        self.concurrency = concurrency
        self.client = client

    def _request(self, conversation: Conversation, sampling: ChatSampling) -> dict[str, Any]:
        system = "\n\n".join(m["content"] for m in conversation if m["role"] == "system")
        messages: list[dict[str, Any]] = []
        for m in conversation:
            if m["role"] == "system":
                continue
            parts = m.get("parts")
            if parts and not messages:
                blocks: list[dict[str, Any]] = [{"type": "text", "text": p} for p in parts if p]
                for block in blocks[-2:]:  # schema, then schema + question: two cache breakpoints
                    block["cache_control"] = {"type": "ephemeral"}
                messages.append({"role": m["role"], "content": blocks})
            else:
                messages.append({"role": m["role"], "content": m["content"]})
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": sampling.max_tokens,
            "messages": messages,
            "cache_control": {"type": "ephemeral"},  # and the growing conversation tail
        }
        if system:
            request["system"] = system
        if sampling.stop:
            request["stop_sequences"] = list(sampling.stop)
        if self.effort:
            request["output_config"] = {"effort": self.effort}
        return request

    def _one(self, conversation: Conversation, sampling: ChatSampling) -> ChatResult:
        start = time.perf_counter()
        with self.client.messages.stream(**self._request(conversation, sampling)) as stream:
            response = stream.get_final_message()
        usage = response.usage
        cache_read = usage.cache_read_input_tokens or 0
        cache_write = usage.cache_creation_input_tokens or 0
        text = "".join(block.text for block in response.content if block.type == "text")
        if response.stop_reason == "refusal":
            text = ""
        return ChatResult(
            text=text,
            finish_reason=response.stop_reason,
            stop_sequence=getattr(response, "stop_sequence", None),
            input_tokens=usage.input_tokens + cache_read + cache_write,
            cached_input_tokens=cache_read,
            cache_write_tokens=cache_write,
            output_tokens=usage.output_tokens,
            latency_s=time.perf_counter() - start,
        )

    def chat(
        self, conversations: Sequence[Conversation], sampling: Sequence[ChatSampling]
    ) -> list[ChatResult]:
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            return list(pool.map(self._one, conversations, sampling))


def create_chat_model(kind: str, model: str, **kwargs: Any) -> ChatModel:
    if kind == "vllm":
        return VLLMChat(model, **kwargs)
    if kind == "openai":
        return OpenAIChat(model, **kwargs)
    if kind == "anthropic":
        return AnthropicChat(model, **kwargs)
    raise ValueError(f"unknown chat backend {kind!r}; expected vllm, openai or anthropic")
