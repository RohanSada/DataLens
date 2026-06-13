"""Pluggable schema-context providers.

Every provider returns a rendered schema string for one ``(db_id, question)``.
To keep the textual format identical across strategies (so what the model sees
at train time matches inference time), all providers convert their selected
tables into ``schema_retrieval`` graph dataclasses and render through the single
shared renderer :func:`formatters.render_schema`.

Strategies
----------
* ``full``        - render every table of the database (lossless; no servers).
* ``retrieved``   - ask the schema_retrieval RAG which tables are relevant, then
                    render those (needs Qdrant + Neo4j running + pre-ingested).
* ``gold_union``  - union the retrieved tables with the tables the gold SQL
                    references, plus ``num_distractors`` random tables. Train-time
                    only; guarantees the answer is reachable while still looking
                    like a noisy retrieval result.
"""
from __future__ import annotations

import logging
import random
from typing import Dict, List, Optional, Sequence

# schema_retrieval flat modules (made importable by this package's __init__).
import graph_store
import schema_loader
from formatters import render_schema

from .config import BuildConfig
from .gold_sql import extract_referenced_tables

logger = logging.getLogger(__name__)

SchemaMap = Dict[str, "schema_loader.DatabaseSchema"]


# ---------------------------------------------------------------------------
# Adapter: schema_loader records -> graph_store dataclasses -> render_schema
# ---------------------------------------------------------------------------
def _to_graph_column(
    column: "schema_loader.ColumnRecord",
    *,
    include_descriptions: bool,
    include_value_hints: bool,
) -> "graph_store.GraphColumn":
    return graph_store.GraphColumn(
        name=column.name,
        friendly_name=column.friendly_name,
        type=column.col_type,
        is_primary_key=column.is_primary_key,
        is_foreign_key=column.is_foreign_key,
        description=column.description if include_descriptions else "",
        value_description=column.value_description if include_value_hints else "",
        sample_values=list(column.sample_values) if include_value_hints else [],
        ordinal=column.ordinal,
    )


def _to_graph_table(
    table: "schema_loader.TableRecord",
    *,
    include_descriptions: bool,
    include_value_hints: bool,
) -> "graph_store.GraphTable":
    return graph_store.GraphTable(
        name=table.name,
        friendly_name=table.friendly_name,
        columns=[
            _to_graph_column(
                col,
                include_descriptions=include_descriptions,
                include_value_hints=include_value_hints,
            )
            for col in table.columns
        ],
    )


def render_tables(
    schema: "schema_loader.DatabaseSchema",
    ordered_names: Optional[Sequence[str]],
    cfg: BuildConfig,
) -> str:
    """Render the given tables (in order) via the shared renderer.

    Args:
        schema: the source database schema (from ``schema_loader``).
        ordered_names: table names to render, in render priority order. ``None``
            renders all tables in their natural schema order.
        cfg: build configuration (controls descriptions / value hints / FK /
            token budget).
    """
    by_name = {t.name: t for t in schema.tables}
    if ordered_names is None:
        chosen = list(schema.tables)
    else:
        # Preserve requested order, skip unknown names, dedupe.
        seen = set()
        chosen = []
        for name in ordered_names:
            table = by_name.get(name)
            if table is not None and name not in seen:
                chosen.append(table)
                seen.add(name)

    gtables = [
        _to_graph_table(
            t,
            include_descriptions=cfg.include_descriptions,
            include_value_hints=cfg.include_value_hints,
        )
        for t in chosen
    ]
    if cfg.include_fk:
        gfks = [
            graph_store.GraphForeignKey(
                from_table=fk.from_table,
                from_column=fk.from_column,
                to_table=fk.to_table,
                to_column=fk.to_column,
            )
            for fk in schema.foreign_keys
        ]
    else:
        gfks = []

    return render_schema(gtables, gfks, token_budget=cfg.schema_token_budget)


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------
class SchemaContextProvider:
    """Base interface: produce a rendered schema string for one example."""

    def context_for(
        self, db_id: str, question: str, evidence: str, gold_sql: str
    ) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - default no-op
        pass


class FullSchemaProvider(SchemaContextProvider):
    """Render the entire database schema. Lossless; requires no servers."""

    def __init__(self, schemas: SchemaMap, cfg: BuildConfig):
        self._schemas = schemas
        self._cfg = cfg

    def context_for(
        self, db_id: str, question: str, evidence: str, gold_sql: str
    ) -> str:
        schema = self._schemas.get(db_id)
        if schema is None:
            raise KeyError(
                f"db_id '{db_id}' not found in loaded schemas "
                f"({sorted(self._schemas)})"
            )
        return render_tables(schema, None, self._cfg)


class RetrievedProvider(SchemaContextProvider):
    """Render the tables the schema_retrieval RAG selects for the question."""

    def __init__(self, schemas: SchemaMap, cfg: BuildConfig, rag):
        self._schemas = schemas
        self._cfg = cfg
        self._rag = rag

    def _selected_tables(self, db_id: str, question: str, evidence: str) -> List[str]:
        result = self._rag.query(question=question, db_id=db_id, evidence=evidence)
        names = [t.name for t in result.tables]
        if not names:
            logger.warning(
                "Retriever returned no tables for db '%s' (question: %.60s). "
                "Was the database ingested?",
                db_id,
                question,
            )
        return names

    def context_for(
        self, db_id: str, question: str, evidence: str, gold_sql: str
    ) -> str:
        schema = self._schemas.get(db_id)
        if schema is None:
            raise KeyError(f"db_id '{db_id}' not found in loaded schemas")
        names = self._selected_tables(db_id, question, evidence)
        return render_tables(schema, names, self._cfg)

    def close(self) -> None:
        try:
            self._rag.close()
        except Exception:  # pragma: no cover - best effort
            pass


class GoldUnionProvider(SchemaContextProvider):
    """Union retrieved tables with gold-required tables, plus distractors.

    Gold tables are rendered first so a token budget can never drop them.
    Requires the gold SQL, so this is a train-time-only strategy.
    """

    def __init__(self, schemas: SchemaMap, cfg: BuildConfig, rag):
        self._schemas = schemas
        self._cfg = cfg
        self._rag = rag
        self._rng = random.Random(cfg.seed)

    def context_for(
        self, db_id: str, question: str, evidence: str, gold_sql: str
    ) -> str:
        schema = self._schemas.get(db_id)
        if schema is None:
            raise KeyError(f"db_id '{db_id}' not found in loaded schemas")
        if not gold_sql or not gold_sql.strip():
            raise ValueError(
                "gold_union strategy requires a gold SQL for every example"
            )

        all_names = [t.name for t in schema.tables]
        gold = extract_referenced_tables(gold_sql, all_names)

        result = self._rag.query(question=question, db_id=db_id, evidence=evidence)
        retrieved = [t.name for t in result.tables]

        # Order: gold first (guaranteed), then retrieved extras, then distractors.
        ordered: List[str] = list(gold)
        seen = set(ordered)
        for name in retrieved:
            if name not in seen:
                ordered.append(name)
                seen.add(name)

        if self._cfg.num_distractors > 0:
            remaining = [n for n in all_names if n not in seen]
            self._rng.shuffle(remaining)
            ordered.extend(remaining[: self._cfg.num_distractors])

        return render_tables(schema, ordered, self._cfg)

    def close(self) -> None:
        try:
            self._rag.close()
        except Exception:  # pragma: no cover - best effort
            pass


# ---------------------------------------------------------------------------
# Loading + factory
# ---------------------------------------------------------------------------
def load_schema_map(settings) -> SchemaMap:
    """Load every BIRD database schema into a ``db_id -> DatabaseSchema`` dict."""
    schemas = schema_loader.load_schemas(settings)
    return {s.db_id: s for s in schemas}


def build_provider(cfg: BuildConfig, schemas: SchemaMap, settings) -> SchemaContextProvider:
    """Construct the provider for the configured strategy.

    For retrieval-based strategies this lazily builds a ``SchemaRAG`` (which
    connects to Qdrant + Neo4j); ``full`` never touches those services.
    """
    if cfg.schema_strategy == "full":
        return FullSchemaProvider(schemas, cfg)

    # retrieved / gold_union both need the RAG pipeline.
    from wrapper import SchemaRAG  # imported lazily; pulls qdrant/neo4j clients

    rag = SchemaRAG(settings)
    if cfg.schema_strategy == "retrieved":
        return RetrievedProvider(schemas, cfg, rag)
    return GoldUnionProvider(schemas, cfg, rag)
