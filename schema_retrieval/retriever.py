"""Query-time schema retriever.

Pipeline:
  1. Embed the question (+ evidence).
  2. Vector search in Qdrant (filtered by db_id) -> seed tables/columns.
  3. Graph expansion in Neo4j -> full columns of seed tables + FK neighbors.
  4. Rank/trim tables to a token budget and render a compact schema context.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config import Settings, get_settings
from embeddings import BaseEmbedder, get_embedder
from formatters import render_schema
from graph_store import GraphForeignKey, GraphStore, GraphTable
from vector_store import GRANULARITY_COLUMN, GRANULARITY_TABLE, VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    db_id: str
    question: str
    context: str  # prompt-ready schema string
    tables: List[GraphTable] = field(default_factory=list)
    foreign_keys: List[GraphForeignKey] = field(default_factory=list)
    seed_tables: List[str] = field(default_factory=list)
    relevant_columns: Dict[str, List[str]] = field(default_factory=dict)


class SchemaRetriever:
    """Retrieves a compact, joinable schema subset for a question."""

    def __init__(
        self,
        settings: Settings | None = None,
        embedder: BaseEmbedder | None = None,
        vector_store: VectorStore | None = None,
        graph_store: GraphStore | None = None,
    ):
        self.settings = settings or get_settings()
        self.embedder = embedder or get_embedder(self.settings)
        self.vector_store = vector_store or VectorStore(self.settings)
        self.graph_store = graph_store or GraphStore(self.settings)

    def retrieve(
        self, question: str, db_id: str, evidence: str = ""
    ) -> RetrievalResult:
        query_text = question if not evidence else f"{question}\n{evidence}"
        query_vector = self.embedder.embed_query(query_text)

        column_hits = self.vector_store.search(
            query_vector,
            db_id=db_id,
            granularity=GRANULARITY_COLUMN,
            limit=self.settings.top_k_columns,
        )
        table_hits = self.vector_store.search(
            query_vector,
            db_id=db_id,
            granularity=GRANULARITY_TABLE,
            limit=self.settings.top_k_tables,
        )

        # Aggregate a relevance score per table from both granularities.
        table_scores: Dict[str, float] = defaultdict(float)
        relevant_columns: Dict[str, List[str]] = defaultdict(list)
        for hit in column_hits:
            if not hit.table:
                continue
            table_scores[hit.table] += hit.score
            if hit.column:
                relevant_columns[hit.table].append(hit.column)
        for hit in table_hits:
            if hit.table:
                table_scores[hit.table] += hit.score

        seed_tables = sorted(
            table_scores, key=lambda t: table_scores[t], reverse=True
        )

        if not seed_tables:
            logger.warning(
                "No vector hits for db '%s'; was it ingested?", db_id
            )
            return RetrievalResult(
                db_id=db_id, question=question, context="", seed_tables=[]
            )

        # Graph expansion: pull FK-connected neighbors so joins are possible.
        expanded = self.graph_store.expand_tables(
            db_id, seed_tables, hops=self.settings.fk_expansion_hops
        )

        # Rank: seed tables first (by score), then neighbors. Cap the count.
        neighbor_tables = [t for t in expanded if t not in table_scores]
        ranked = seed_tables + sorted(neighbor_tables)
        selected = ranked[: self.settings.max_tables_in_context]

        tables = self.graph_store.fetch_tables(db_id, selected)
        # Preserve ranked ordering for rendering priority.
        order = {name: i for i, name in enumerate(selected)}
        tables.sort(key=lambda t: order.get(t.name, len(order)))
        foreign_keys = self.graph_store.fetch_foreign_keys(db_id, selected)

        highlight = [c for cols in relevant_columns.values() for c in cols]
        context = render_schema(
            tables,
            foreign_keys,
            highlight_columns=highlight,
            token_budget=self.settings.token_budget,
        )

        return RetrievalResult(
            db_id=db_id,
            question=question,
            context=context,
            tables=tables,
            foreign_keys=foreign_keys,
            seed_tables=seed_tables,
            relevant_columns=dict(relevant_columns),
        )

    def close(self) -> None:
        self.vector_store.close()
        self.graph_store.close()
