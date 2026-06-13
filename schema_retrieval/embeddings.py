"""Pluggable text embedding backends.

The local provider (sentence-transformers) is the default: free, private and
offline. An OpenAI provider is available when higher-quality embeddings are
desired. Both expose the same interface so the rest of the system is agnostic to
the choice.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Sequence

from config import Settings, get_settings

logger = logging.getLogger(__name__)

# Known output dimensions for common OpenAI models (avoids a probe call).
_OPENAI_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class BaseEmbedder(ABC):
    """Common interface for embedding providers."""

    dim: int

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed a batch of documents (schema records)."""

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single search query."""


class LocalEmbedder(BaseEmbedder):
    """sentence-transformers backed embedder with cosine-normalized output."""

    def __init__(self, model_name: str, batch_size: int = 64):
        # Imported lazily so the package is importable without torch installed.
        from sentence_transformers import SentenceTransformer

        logger.info("Loading local embedding model: %s", model_name)
        self._model = SentenceTransformer(model_name)
        self._batch_size = batch_size
        self.dim = int(self._model.get_sentence_embedding_dimension())

        # BGE retrieval models expect an instruction prefix on the *query* side.
        self._query_instruction = (
            "Represent this sentence for searching relevant passages: "
            if "bge" in model_name.lower()
            else ""
        )

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        payload = f"{self._query_instruction}{text}"
        vector = self._model.encode(
            payload,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vector.tolist()


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI embeddings backend."""

    def __init__(self, model_name: str, api_key: str | None, batch_size: int = 64):
        from openai import OpenAI

        if not api_key:
            raise ValueError(
                "OpenAI embedding provider selected but no API key configured "
                "(set SCHEMA_RAG_OPENAI_API_KEY)."
            )
        self._client = OpenAI(api_key=api_key)
        self._model_name = model_name
        self._batch_size = batch_size
        self.dim = _OPENAI_DIMS.get(model_name) or len(
            self._embed_raw(["dimension probe"])[0]
        )

    def _embed_raw(self, texts: Sequence[str]) -> List[List[float]]:
        response = self._client.embeddings.create(
            model=self._model_name, input=list(texts)
        )
        return [item.embedding for item in response.data]

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        out: List[List[float]] = []
        for start in range(0, len(texts), self._batch_size):
            out.extend(self._embed_raw(texts[start : start + self._batch_size]))
        return out

    def embed_query(self, text: str) -> List[float]:
        return self._embed_raw([text])[0]


def get_embedder(settings: Settings | None = None) -> BaseEmbedder:
    """Factory that returns the configured embedder."""
    settings = settings or get_settings()
    if settings.embedding_provider == "openai":
        return OpenAIEmbedder(
            model_name=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
            batch_size=settings.embedding_batch_size,
        )
    return LocalEmbedder(
        model_name=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
    )
