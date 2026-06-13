"""Configuration for building BIRD Text-to-SQL training data."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

SchemaStrategy = Literal["full", "retrieved", "gold_union"]


@dataclass
class BuildConfig:
    """All knobs for a single dataset build.

    Attributes:
        input_path: BIRD-format JSON (list of objects with ``db_id``,
            ``question``, ``evidence``, ``SQL`` keys; ``difficulty`` optional).
        out_dir: directory where ``train.jsonl`` (and ``val.jsonl``) are written.
        schema_strategy: how the per-example schema context is built.
            ``full`` renders the entire DB schema (lossless, no servers needed).
            ``retrieved`` uses the schema_retrieval RAG pipeline (needs Qdrant +
            Neo4j running and pre-ingested). ``gold_union`` unions the retrieved
            tables with the tables referenced by the gold SQL plus distractors
            (train-time only; guarantees answerability while mimicking retrieval).
        include_descriptions / include_value_hints / include_fk: toggle the
            corresponding parts of the rendered schema.
        num_distractors: extra random tables added by ``gold_union``.
        schema_token_budget: optional ~token cap for the rendered schema
            (``None`` = no truncation; recommended for train-time coverage).
        val_split: fraction held out into ``val.jsonl`` (0.0 = single file).
        seed: RNG seed for the split and distractor sampling.
        db_filter: if set, only emit examples for this ``db_id``.
        settings_path: path to the schema_retrieval ``settings.json`` (defaults
            to the one shipped next to that package).
        report_coverage: parse gold SQL and report table-coverage of the
            produced contexts (validates full/gold_union, measures retrieved
            recall).
        limit: optional cap on the number of examples processed (debugging).
    """

    input_path: Path
    out_dir: Path
    schema_strategy: SchemaStrategy = "full"
    include_descriptions: bool = True
    include_value_hints: bool = True
    include_fk: bool = True
    num_distractors: int = 0
    schema_token_budget: Optional[int] = None
    val_split: float = 0.0
    seed: int = 42
    db_filter: Optional[str] = None
    settings_path: Optional[Path] = None
    report_coverage: bool = False
    limit: Optional[int] = None

    def __post_init__(self) -> None:
        self.input_path = Path(self.input_path)
        self.out_dir = Path(self.out_dir)
        if self.settings_path is not None:
            self.settings_path = Path(self.settings_path)
        if self.schema_strategy not in ("full", "retrieved", "gold_union"):
            raise ValueError(
                f"Unknown schema_strategy '{self.schema_strategy}'; "
                "expected one of: full, retrieved, gold_union"
            )
        if not 0.0 <= self.val_split < 1.0:
            raise ValueError("val_split must be in [0.0, 1.0)")

    @property
    def needs_retriever(self) -> bool:
        """True when the strategy requires the schema_retrieval RAG stores."""
        return self.schema_strategy in ("retrieved", "gold_union")
