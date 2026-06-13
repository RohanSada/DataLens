"""High-level, user-facing wrapper around the schema RAG pipeline.

:class:`SchemaRAG` is the single entry point. Initialize it from a
``settings.json`` file (which controls all paths, database connections,
embedding and retrieval options), then use it to ingest schemas and query them.

Two ways to load data:

* :meth:`ingest` -- bulk-ingest the whole BIRD dataset referenced by the
  settings file (``bird_root`` / ``dev_tables.json``).
* :meth:`build` -- ingest your own ``.sqlite`` + schema ``.json``.

Example::

    rag = SchemaRAG("settings.json")
    rag.ingest(rebuild=True)                 # load the BIRD dataset
    result = rag.query("How many active users signed up last month?",
                       db_id="financial")
    print(result.context)
    rag.close()
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from config import Settings, get_settings, load_settings
from embeddings import BaseEmbedder, get_embedder
from graph_store import GraphStore
from retriever import RetrievalResult, SchemaRetriever
from schema_loader import DatabaseSchema, load_schema_from_files, load_schemas
from vector_store import VectorStore

logger = logging.getLogger(__name__)


class SchemaRAG:
    """One-stop wrapper to build schema stores and query them.

    All heavy components (embedder, vector store, graph store) are created once
    and shared between ingestion and retrieval so resources are reused.
    """

    def __init__(
        self,
        settings: Settings | str | Path | None = None,
        embedder: BaseEmbedder | None = None,
        vector_store: VectorStore | None = None,
        graph_store: GraphStore | None = None,
    ):
        """Initialize the pipeline from a ``settings.json`` file.

        Args:
            settings: a path to a ``settings.json`` file, a pre-built
                :class:`Settings` instance, or ``None`` to use the default
                ``settings.json`` next to the package.
        """
        if isinstance(settings, (str, Path)):
            self.settings = load_settings(settings)
        elif settings is None:
            self.settings = get_settings()
        else:
            self.settings = settings
        self.embedder = embedder or get_embedder(self.settings)
        self.vector_store = vector_store or VectorStore(self.settings)
        self.graph_store = graph_store or GraphStore(self.settings)
        self._retriever = SchemaRetriever(
            settings=self.settings,
            embedder=self.embedder,
            vector_store=self.vector_store,
            graph_store=self.graph_store,
        )
        # db_ids built in this session (used to default the query target).
        self._db_ids: List[str] = []

    # ------------------------------------------------------------------
    # Ingest (bulk BIRD dataset)
    # ------------------------------------------------------------------
    def ingest(
        self,
        db_id: Optional[str] = None,
        rebuild: bool = False,
    ) -> dict:
        """Bulk-ingest the BIRD dataset referenced by the settings file.

        Loads every database (or a single one when ``db_id`` is given) from the
        ``dev_tables.json`` resolved via the settings, then populates the vector
        (Qdrant) and graph (Neo4j) stores.

        Args:
            db_id: ingest only this database; default ingests all of them.
            rebuild: drop the vector collection and clear graph data first for a
                clean slate (only clears the whole graph when ``db_id`` is None).

        Returns:
            A summary dict with the built ``db_ids``, table and point counts.
        """
        schemas: List[DatabaseSchema] = load_schemas(self.settings)
        if db_id:
            schemas = [s for s in schemas if s.db_id == db_id]
            if not schemas:
                raise ValueError(
                    f"No database named '{db_id}' found in {self.settings.tables_json_path}"
                )

        self.vector_store.ensure_collection(dim=self.embedder.dim, recreate=rebuild)
        self.graph_store.ensure_constraints()
        if rebuild and not db_id:
            self.graph_store.clear()

        total_points = 0
        total_tables = 0
        built: List[str] = []
        for schema in schemas:
            total_points += self.vector_store.upsert_schema(schema, self.embedder)
            self.graph_store.ingest_schema(schema)
            total_tables += len(schema.tables)
            built.append(schema.db_id)
            if schema.db_id not in self._db_ids:
                self._db_ids.append(schema.db_id)

        summary = {
            "db_ids": built,
            "tables": total_tables,
            "points": total_points,
        }
        logger.info("Ingest complete: %s", summary)
        return summary

    # ------------------------------------------------------------------
    # Build (user-supplied files)
    # ------------------------------------------------------------------
    def build(
        self,
        schema_path: Path | str,
        db_path: Optional[Path | str] = None,
        db_id: Optional[str] = None,
        description_dir: Optional[Path | str] = None,
        rebuild: bool = False,
    ) -> dict:
        """Construct the vector + graph stores from a user DB + schema JSON.

        Args:
            db_path: path to the ``.sqlite`` file (used for value sampling).
            schema_path: path to the schema JSON (BIRD ``dev_tables.json`` format).
            db_id: required only when the JSON contains multiple databases.
            description_dir: optional folder of ``<table>.csv`` description files.
            rebuild: drop the database's existing points/nodes before ingesting.

        Returns:
            A summary dict with the built ``db_ids``, table and point counts.
        """
        schemas: List[DatabaseSchema] = load_schema_from_files(
            schema_json=schema_path,
            sqlite_path=db_path,
            db_id=db_id,
            description_dir=description_dir,
            settings=self.settings,
        )

        self.vector_store.ensure_collection(dim=self.embedder.dim, recreate=False)
        self.graph_store.ensure_constraints()

        total_points = 0
        total_tables = 0
        built: List[str] = []
        for schema in schemas:
            if rebuild:
                self.vector_store.delete_database(schema.db_id)
                self.graph_store.clear_database(schema.db_id)
            total_points += self.vector_store.upsert_schema(schema, self.embedder)
            self.graph_store.ingest_schema(schema)
            total_tables += len(schema.tables)
            built.append(schema.db_id)
            if schema.db_id not in self._db_ids:
                self._db_ids.append(schema.db_id)

        summary = {
            "db_ids": built,
            "tables": total_tables,
            "points": total_points,
        }
        logger.info("Build complete: %s", summary)
        return summary

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def query(
        self,
        question: str,
        db_id: Optional[str] = None,
        evidence: str = "",
    ) -> RetrievalResult:
        """Retrieve the relevant schema subset for a natural-language question.

        Returns the full :class:`RetrievalResult`: relevant tables, columns,
        foreign-key join paths, value hints, and a prompt-ready ``context``
        string.
        """
        target = db_id or self._default_db_id()
        return self._retriever.retrieve(
            question=question, db_id=target, evidence=evidence
        )

    def _default_db_id(self) -> str:
        if len(self._db_ids) == 1:
            return self._db_ids[0]
        if not self._db_ids:
            raise ValueError(
                "No database has been built in this session; pass db_id "
                "explicitly or call build() first."
            )
        raise ValueError(
            f"Multiple databases are available {self._db_ids}; specify db_id."
        )

    def close(self) -> None:
        self.vector_store.close()
        self.graph_store.close()

    def __enter__(self) -> "SchemaRAG":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    rag = SchemaRAG("settings.json")
    rag.ingest(db_id="debit_card_specializing", rebuild=True)

    result = rag.query(
        "In 2012, who had the least consumption in LAM?",
        db_id="debit_card_specializing",
    )
    print(result.context)
    rag.close()

