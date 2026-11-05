"""The models behind the agent's roles, loaded lazily and freed between stages."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from datalens.agent.config import ModelEntry, RoleSpec
from datalens.inference.backends import Backend, Generation, SamplingConfig, create_backend
from datalens.prompts import PromptParts

log = logging.getLogger(__name__)


class RoleModel:
    """One role's handle on a backend, with its LoRA adapter bound."""

    def __init__(
        self, role: str, backend: Backend, *, lora: str | None = None, max_model_len: int | None = None
    ) -> None:
        self.role = role
        self.backend = backend
        self.lora = lora
        self.max_model_len = max_model_len

    @property
    def name(self) -> str:
        return f"{self.backend.name}+{self.lora}" if self.lora else self.backend.name

    def generate(self, prompts: Sequence[PromptParts], sampling: SamplingConfig) -> list[Generation]:
        if not prompts:
            return []
        if self.lora:
            generate: Callable[..., list[Generation]] = self.backend.generate
            return generate(prompts, sampling, lora_path=self.lora)
        return self.backend.generate(prompts, sampling)

    def fits(self, prompt: PromptParts, max_tokens: int) -> bool:
        """Whether ``prompt`` plus ``max_tokens`` fits the context (vLLM rejects longer prompts)."""
        if not self.max_model_len:
            return True
        chars = len(prompt.system) + len(prompt.user)
        if chars / 2 + max_tokens < self.max_model_len:
            return True  # even at 2 characters per token it fits; skip tokenizing
        count = getattr(self.backend, "count_tokens", None)
        tokens = count(prompt) if callable(count) else chars / 3
        return bool(tokens + max_tokens <= self.max_model_len)


class ModelPool:
    """Creates backends on first use; ``release`` frees the ones no longer needed."""

    def __init__(
        self,
        models: Mapping[str, ModelEntry],
        *,
        backends: Mapping[str, Backend] | None = None,
        factory: Callable[..., Backend] | None = None,
        max_model_len: int | None = None,
    ) -> None:
        self.entries = dict(models)
        self._live: dict[str, Backend] = dict(backends or {})
        self._injected = set(self._live)
        self._factory = factory
        self.max_model_len = max_model_len

    def backend(self, name: str) -> Backend:
        if name not in self._live:
            entry = self.entries[name]
            log.info("loading %s (%s)", entry.model, entry.backend)
            factory = self._factory or create_backend
            self._live[name] = factory(entry.backend, entry.model, **dict(entry.backend_kwargs))
        return self._live[name]

    def role(self, role: str, spec: RoleSpec) -> RoleModel:
        entry = self.entries.get(spec.model)
        limit = self.max_model_len
        if limit is None and entry is not None:
            limit = entry.backend_kwargs.get("max_model_len")
        return RoleModel(role, self.backend(spec.model), lora=spec.lora, max_model_len=limit)

    def loaded(self) -> list[str]:
        return list(self._live)

    def release(self, keep: Iterable[str] = ()) -> None:
        """Close every loaded backend not in ``keep`` (injected test backends stay)."""
        keep = set(keep) | self._injected
        for name in [n for n in self._live if n not in keep]:
            backend = self._live.pop(name)
            close: Any = getattr(backend, "close", None)
            if callable(close):
                log.info("releasing %s", name)
                close()
