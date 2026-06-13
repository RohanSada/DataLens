"""Qdrant vector store wrapper.

Stores two granularities of points - ``column`` and ``table`` - all in a single
collection, with payload indexes that make ``db_id``-filtered search cheap even
as the number of indexed databases grows.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from config import Settings, get_settings
from schema_loader import DatabaseSchema

logger = logging.getLogger(__name__)

# Stable namespace so point IDs are deterministic across re-ingests.
_ID_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")

GRANULARITY_COLUMN = "column"
GRANULARITY_TABLE = "table"


def make_point_id(db_id: str, table: str, granularity: str, column: str = "") -> str:
    key = f"{granularity}|{db_id}|{table}|{column}"
    return str(uuid.uuid5(_ID_NAMESPACE, key))


@dataclass
class SchemaHit:
    score: float
    granularity: str
    db_id: str
    table: str
    column: Optional[str]
    payload: dict


class VectorStore:
    """Thin, production-minded wrapper around the Qdrant client."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.collection = self.settings.qdrant_collection
        self.client = QdrantClient(
            url=self.settings.qdrant_url,
            api_key=self.settings.qdrant_api_key,
            timeout=60,
        )

    # ------------------------------------------------------------------
    # Collection lifecycle
    # ------------------------------------------------------------------
    def ensure_collection(self, dim: int, recreate: bool = False) -> None:
        exists = self.client.collection_exists(self.collection)
        if exists and recreate:
            logger.info("Dropping existing collection %s", self.collection)
            self.client.delete_collection(self.collection)
            exists = False
        if not exists:
            logger.info("Creating collection %s (dim=%d)", self.collection, dim)
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=qmodels.VectorParams(
                    size=dim, distance=qmodels.Distance.COSINE
                ),
            )
            self._create_payload_indexes()

    def _create_payload_indexes(self) -> None:
        for field_name in ("db_id", "table", "granularity"):
            try:
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field_name,
                    field_schema=qmodels.PayloadSchemaType.KEYWORD,
                )
            except Exception as exc:  # index may already exist
                logger.debug("Payload index for %s: %s", field_name, exc)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def upsert_schema(
        self,
        schema: DatabaseSchema,
        embedder,
        batch_size: int = 128,
    ) -> int:
        """Embed and upsert all columns + tables for one database."""
        texts: List[str] = []
        points_meta: List[dict] = []

        for table in schema.tables:
            points_meta.append(
                {
                    "id": make_point_id(schema.db_id, table.name, GRANULARITY_TABLE),
                    "payload": {
                        "granularity": GRANULARITY_TABLE,
                        "db_id": schema.db_id,
                        "table": table.name,
                        "column": None,
                        "friendly_name": table.friendly_name,
                    },
                }
            )
            texts.append(table.embedding_text())

            for column in table.columns:
                points_meta.append(
                    {
                        "id": make_point_id(
                            schema.db_id,
                            table.name,
                            GRANULARITY_COLUMN,
                            column.name,
                        ),
                        "payload": {
                            "granularity": GRANULARITY_COLUMN,
                            "db_id": schema.db_id,
                            "table": table.name,
                            "column": column.name,
                            "friendly_name": column.friendly_name,
                            "type": column.col_type,
                            "is_primary_key": column.is_primary_key,
                            "is_foreign_key": column.is_foreign_key,
                            "description": column.description,
                            "value_description": column.value_description,
                        },
                    }
                )
                texts.append(column.embedding_text())

        vectors = embedder.embed_documents(texts)
        '''
        points = [
            qmodels.PointStruct(
                id=meta["id"], vector=vector, payload=meta["payload"]
            )
            for meta, vector in zip(points_meta, vectors)
        ]
        '''
        points = []
        for meta, vector in zip(points_meta, vectors):
            points.append(
                qmodels.PointStruct(id=meta["id"], vector=vector, payload=meta["payload"])
            )

        for start in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.collection,
                points=points[start : start + batch_size],
                wait=True,
            )
        logger.info(
            "Upserted %d points for db '%s'", len(points), schema.db_id
        )
        return len(points)

    def delete_database(self, db_id: str) -> None:
        """Remove all points for a single database (for clean per-DB rebuilds)."""
        if not self.client.collection_exists(self.collection):
            return
        self.client.delete(
            collection_name=self.collection,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="db_id", match=qmodels.MatchValue(value=db_id)
                        )
                    ]
                )
            ),
            wait=True,
        )
        logger.info("Deleted existing vector points for db '%s'", db_id)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(
        self,
        query_vector: Sequence[float],
        db_id: str,
        granularity: Optional[str] = None,
        limit: int = 25,
    ) -> List[SchemaHit]:
        must: List[qmodels.FieldCondition] = [
            qmodels.FieldCondition(
                key="db_id", match=qmodels.MatchValue(value=db_id)
            )
        ]
        if granularity:
            must.append(
                qmodels.FieldCondition(
                    key="granularity",
                    match=qmodels.MatchValue(value=granularity),
                )
            )

        response = self.client.query_points(
            collection_name=self.collection,
            query=list(query_vector),
            query_filter=qmodels.Filter(must=must),
            limit=limit,
            with_payload=True,
        )
        hits: List[SchemaHit] = []
        for point in response.points:
            payload = point.payload or {}
            hits.append(
                SchemaHit(
                    score=point.score,
                    granularity=payload.get("granularity", ""),
                    db_id=payload.get("db_id", db_id),
                    table=payload.get("table", ""),
                    column=payload.get("column"),
                    payload=payload,
                )
            )
        return hits

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:  # pragma: no cover - best effort
            pass
