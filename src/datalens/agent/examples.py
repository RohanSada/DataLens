"""Few-shot examples for the ICL generator, retrieved by question skeleton.

Agentar-Scale-SQL (like DAIL-SQL) masks a question's database literals and
column names to get its *skeleton*, e.g. "In <MONTH> of <YEAR>, how many
<COLUMN> were placed by the <COLUMN> with the highest amount of <COLUMN>?",
embeds it, and retrieves training questions with similar skeletons as
demonstrations. Examples then match the structure of the question rather than
its topic, and come from other databases, so nothing about the test database
leaks in.

Skeletons come from the task-understanding model (the paper's choice) or from
:func:`heuristic_skeleton`. Embeddings come from a sentence-transformers model
(the paper uses all-MiniLM-L6-v2) or, when that library is not installed, from
a TF-IDF index, which needs nothing beyond numpy.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

STORE_VERSION = 1
_WORD_RE = re.compile(r"<[A-Z]+>|[0-9a-z]+")


@dataclass(frozen=True)
class Demo:
    """One training question shown to the ICL generator as an example."""

    id: str
    db_id: str
    question: str
    evidence: str
    sql: str
    skeleton: str


# ------------------------------------------------------------------ skeletons

_QUOTED_RE = re.compile(r"(['\"`])(?:(?!\1).){1,80}\1")
_YEAR_RE = re.compile(r"\b(?:1[5-9]|20)\d{2}\b")
_NUMBER_RE = re.compile(r"(?<![<\w])[-+]?\d+(?:[.,]\d+)*%?(?![\w>])")
_SENTENCE_END = (".", "?", "!", ":", ";")


def heuristic_skeleton(question: str, columns: Iterable[str] = ()) -> str:
    """Mask literals and column names without a model.

    Quoted strings become ``<VALUE>``, years ``<YEAR>``, other numbers
    ``<NUMBER>``, capitalised words that do not start a sentence ``<VALUE>``
    and mentions of schema column names ``<COLUMN>``.
    """
    text = " ".join(question.split())
    text = _QUOTED_RE.sub("<VALUE>", text)
    text = _YEAR_RE.sub("<YEAR>", text)
    text = _NUMBER_RE.sub("<NUMBER>", text)

    names = sorted(
        {" ".join(c.replace("_", " ").split()).lower() for c in columns if len(c) > 2},
        key=len,
        reverse=True,
    )
    for name in names:
        text = re.sub(rf"(?<![\w<]){re.escape(name)}(?![\w>])", "<COLUMN>", text, flags=re.IGNORECASE)

    words = text.split(" ")
    out: list[str] = []
    for i, word in enumerate(words):
        core = word.strip(",.?!:;()[]")
        starts_sentence = i == 0 or words[i - 1].endswith(_SENTENCE_END)
        if core and core[0].isupper() and not core.startswith("<") and not starts_sentence and core != "I":
            word = word.replace(core, "<VALUE>")
        if word == "<VALUE>" and out and out[-1] == "<VALUE>":
            continue  # "New York City" is one value
        out.append(word)
    return " ".join(out)


# ------------------------------------------------------------------ embeddings


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(re.sub(r"<([A-Za-z]+)>", lambda m: f"<{m.group(1).upper()}>", text).lower())


class TfidfIndex:
    """Sparse TF-IDF cosine similarity over a fixed set of texts."""

    def __init__(self, texts: Sequence[str]) -> None:
        docs = [Counter(_words(t)) for t in texts]
        df: Counter[str] = Counter()
        for doc in docs:
            df.update(doc.keys())
        n = max(len(docs), 1)
        self.idf = {w: math.log((1 + n) / (1 + c)) + 1.0 for w, c in df.items()}
        self.postings: dict[str, list[tuple[int, float]]] = {}
        for i, doc in enumerate(docs):
            weights = {w: (1 + math.log(c)) * self.idf[w] for w, c in doc.items()}
            norm = math.sqrt(sum(v * v for v in weights.values())) or 1.0
            for w, v in weights.items():
                self.postings.setdefault(w, []).append((i, v / norm))
        self.size = len(docs)

    def scores(self, text: str) -> np.ndarray:
        counts = Counter(w for w in _words(text) if w in self.idf)
        weights = {w: (1 + math.log(c)) * self.idf[w] for w, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in weights.values())) or 1.0
        out = np.zeros(self.size, dtype=np.float32)
        for w, v in weights.items():
            for i, dv in self.postings[w]:
                out[i] += dv * v / norm
        return out


class SentenceEmbedder:
    """Normalised sentence embeddings from a sentence-transformers model."""

    def __init__(self, model_name: str, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer  # optional dependency

        self.name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self.model.encode(
            list(texts), batch_size=256, normalize_embeddings=True, convert_to_numpy=True
        )
        return np.asarray(vectors, dtype=np.float32)


def make_embedder(model_name: str | None) -> SentenceEmbedder | None:
    """A sentence embedder, or ``None`` (TF-IDF) if the model or library is unavailable."""
    if not model_name:
        return None
    try:
        return SentenceEmbedder(model_name)
    except ImportError:
        log.warning("sentence-transformers is not installed; retrieving examples with TF-IDF")
        return None


# ------------------------------------------------------------------ store


class ExampleStore:
    """Training questions searchable by skeleton similarity."""

    def __init__(
        self,
        demos: list[Demo],
        *,
        vectors: np.ndarray | None = None,
        embedder: SentenceEmbedder | None = None,
        embedder_name: str | None = None,
    ) -> None:
        if vectors is not None and len(vectors) != len(demos):
            raise ValueError("one vector per demo is required")
        self.demos = demos
        self.vectors = vectors
        self.embedder = embedder
        self.embedder_name = embedder_name or (embedder.name if embedder else None)
        self._tfidf = TfidfIndex([d.skeleton for d in demos]) if vectors is None else None

    @classmethod
    def build(cls, demos: list[Demo], embedder: SentenceEmbedder | None = None) -> ExampleStore:
        vectors = embedder.encode([d.skeleton for d in demos]) if embedder is not None and demos else None
        return cls(demos, vectors=vectors, embedder=embedder)

    def __len__(self) -> int:
        return len(self.demos)

    def _scores(self, skeletons: Sequence[str]) -> np.ndarray:
        if self.vectors is not None:
            if self.embedder is None:
                raise RuntimeError(
                    f"this store was embedded with {self.embedder_name}; load it with that embedder"
                )
            return self.embedder.encode(skeletons) @ self.vectors.T
        assert self._tfidf is not None
        return np.stack([self._tfidf.scores(s) for s in skeletons]) if skeletons else np.zeros((0, 0))

    def search_many(
        self,
        skeletons: Sequence[str],
        k: int,
        *,
        exclude_dbs: Sequence[str | None] | None = None,
    ) -> list[list[Demo]]:
        """Top-``k`` demos per skeleton, skipping demos from ``exclude_dbs[i]``."""
        if k <= 0 or not self.demos or not skeletons:
            return [[] for _ in skeletons]
        scores = self._scores(skeletons)
        out = []
        for i, row in enumerate(scores):
            banned = exclude_dbs[i] if exclude_dbs is not None else None
            picked: list[Demo] = []
            for j in np.argsort(-row, kind="stable"):
                demo = self.demos[int(j)]
                if banned is not None and demo.db_id == banned:
                    continue
                picked.append(demo)
                if len(picked) == k:
                    break
            out.append(picked)
        return out

    def by_id(self) -> dict[str, Demo]:
        return {d.id: d for d in self.demos}

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        meta: dict[str, Any] = {"version": STORE_VERSION, "embedder": self.embedder_name}
        (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        with (directory / "demos.jsonl").open("w", encoding="utf-8") as fh:
            for demo in self.demos:
                fh.write(json.dumps(asdict(demo), ensure_ascii=False) + "\n")
        if self.vectors is not None:
            np.save(directory / "vectors.npy", self.vectors)

    @classmethod
    def load(cls, directory: str | Path, embedder: SentenceEmbedder | None = None) -> ExampleStore:
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        if meta.get("version") != STORE_VERSION:
            raise ValueError(f"{directory}: example store version {meta.get('version')}")
        demos = [
            Demo(**json.loads(line))
            for line in (directory / "demos.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        vectors_file = directory / "vectors.npy"
        name = meta.get("embedder")
        if vectors_file.is_file():
            if embedder is None or embedder.name != name:
                embedder = make_embedder(name)
            if embedder is None:
                log.warning("cannot load embedder %s; falling back to TF-IDF retrieval", name)
                return cls(demos)
            return cls(demos, vectors=np.load(vectors_file), embedder=embedder)
        return cls(demos)

    @staticmethod
    def exists(directory: str | Path) -> bool:
        return (Path(directory) / "meta.json").is_file()
