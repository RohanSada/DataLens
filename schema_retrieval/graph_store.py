"""Neo4j graph store wrapper.

Models the schema as Database -> Table -> Column with REFERENCES edges between
columns derived from foreign keys. The graph's job at query time is *expansion*:
turn a handful of semantically-matched columns into a set of fully-described,
joinable tables (with the foreign-key paths that connect them).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

from neo4j import GraphDatabase

from config import Settings, get_settings
from schema_loader import DatabaseSchema

logger = logging.getLogger(__name__)


@dataclass
class GraphColumn:
    name: str
    friendly_name: str
    type: str
    is_primary_key: bool
    is_foreign_key: bool
    description: str
    value_description: str
    sample_values: List[str]
    ordinal: int


@dataclass
class GraphTable:
    name: str
    friendly_name: str
    columns: List[GraphColumn] = field(default_factory=list)


@dataclass
class GraphForeignKey:
    from_table: str
    from_column: str
    to_table: str
    to_column: str


class GraphStore:
    """Thin wrapper around the Neo4j driver."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._driver = GraphDatabase.driver(
            self.settings.neo4j_uri,
            auth=(self.settings.neo4j_user, self.settings.neo4j_password),
        )
        self._database = self.settings.neo4j_database

    # ------------------------------------------------------------------
    # Schema setup
    # ------------------------------------------------------------------
    def ensure_constraints(self) -> None:
        statements = [
            "CREATE CONSTRAINT db_name IF NOT EXISTS "
            "FOR (d:Database) REQUIRE d.name IS UNIQUE",
            "CREATE CONSTRAINT table_uid IF NOT EXISTS "
            "FOR (t:Table) REQUIRE t.uid IS UNIQUE",
            "CREATE CONSTRAINT column_uid IF NOT EXISTS "
            "FOR (c:Column) REQUIRE c.uid IS UNIQUE",
        ]
        with self._driver.session(database=self._database) as session:
            for stmt in statements:
                session.run(stmt)

    def clear(self) -> None:
        """Remove all schema nodes (used by --rebuild)."""
        logger.info("Clearing existing graph data")
        with self._driver.session(database=self._database) as session:
            session.run(
                "MATCH (n) WHERE n:Database OR n:Table OR n:Column "
                "DETACH DELETE n"
            )

    def clear_database(self, db_id: str) -> None:
        """Remove a single database's nodes (for clean per-DB rebuilds)."""
        logger.info("Clearing existing graph data for db '%s'", db_id)
        with self._driver.session(database=self._database) as session:
            session.run(
                """
                MATCH (d:Database {name: $db_id})
                OPTIONAL MATCH (d)-[:HAS_TABLE]->(t:Table)
                OPTIONAL MATCH (t)-[:HAS_COLUMN]->(c:Column)
                DETACH DELETE d, t, c
                """,
                db_id=db_id,
            )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def ingest_schema(self, schema: DatabaseSchema) -> None:
        table_rows = []
        column_rows = []
        for table in schema.tables:
            t_uid = f"{schema.db_id}.{table.name}"
            table_rows.append(
                {
                    "uid": t_uid,
                    "name": table.name,
                    "friendly_name": table.friendly_name,
                }
            )
            for column in table.columns:
                column_rows.append(
                    {
                        "uid": f"{t_uid}.{column.name}",
                        "table_uid": t_uid,
                        "name": column.name,
                        "friendly_name": column.friendly_name,
                        "type": column.col_type,
                        "is_primary_key": column.is_primary_key,
                        "is_foreign_key": column.is_foreign_key,
                        "description": column.description,
                        "value_description": column.value_description,
                        "sample_values": column.sample_values,
                        "ordinal": column.ordinal,
                    }
                )

        fk_rows = [
            {
                "from_uid": f"{schema.db_id}.{fk.from_table}.{fk.from_column}",
                "to_uid": f"{schema.db_id}.{fk.to_table}.{fk.to_column}",
            }
            for fk in schema.foreign_keys
        ]

        with self._driver.session(database=self._database) as session:
            session.execute_write(
                self._write_schema, schema.db_id, table_rows, column_rows, fk_rows
            )
        logger.info(
            "Ingested graph for db '%s': %d tables, %d columns, %d FKs",
            schema.db_id,
            len(table_rows),
            len(column_rows),
            len(fk_rows),
        )

    @staticmethod
    def _write_schema(tx, db_id, table_rows, column_rows, fk_rows) -> None:
        tx.run("MERGE (d:Database {name: $db_id})", db_id=db_id)
        tx.run(
            """
            UNWIND $rows AS row
            MATCH (d:Database {name: $db_id})
            MERGE (t:Table {uid: row.uid})
            SET t.name = row.name, t.friendly_name = row.friendly_name,
                t.db_id = $db_id
            MERGE (d)-[:HAS_TABLE]->(t)
            """,
            rows=table_rows,
            db_id=db_id,
        )
        tx.run(
            """
            UNWIND $rows AS row
            MATCH (t:Table {uid: row.table_uid})
            MERGE (c:Column {uid: row.uid})
            SET c.name = row.name, c.friendly_name = row.friendly_name,
                c.type = row.type, c.is_primary_key = row.is_primary_key,
                c.is_foreign_key = row.is_foreign_key,
                c.description = row.description,
                c.value_description = row.value_description,
                c.sample_values = row.sample_values, c.ordinal = row.ordinal,
                c.db_id = $db_id
            MERGE (t)-[:HAS_COLUMN]->(c)
            """,
            rows=column_rows,
            db_id=db_id,
        )
        if fk_rows:
            tx.run(
                """
                UNWIND $rows AS row
                MATCH (a:Column {uid: row.from_uid})
                MATCH (b:Column {uid: row.to_uid})
                MERGE (a)-[:REFERENCES]->(b)
                """,
                rows=fk_rows,
            )

    # ------------------------------------------------------------------
    # Query-time expansion
    # ------------------------------------------------------------------
    def expand_tables(
        self, db_id: str, seed_tables: Sequence[str], hops: int = 1
    ) -> Set[str]:
        """Return seed tables plus FK-connected neighbors up to ``hops``."""
        result: Set[str] = set(seed_tables)
        frontier: Set[str] = set(seed_tables)
        for _ in range(max(0, hops)):
            if not frontier:
                break
            neighbors = self._fk_neighbors(db_id, list(frontier))
            new = neighbors - result
            result.update(neighbors)
            frontier = new
        return result

    def _fk_neighbors(self, db_id: str, tables: List[str]) -> Set[str]:
        query = """
        MATCH (d:Database {name: $db_id})-[:HAS_TABLE]->(t:Table)
              -[:HAS_COLUMN]->(c:Column)
        WHERE t.name IN $tables
        MATCH (c)-[:REFERENCES]-(c2:Column)<-[:HAS_COLUMN]-(t2:Table)
              <-[:HAS_TABLE]-(d)
        RETURN DISTINCT t2.name AS name
        """
        with self._driver.session(database=self._database) as session:
            records = session.run(query, db_id=db_id, tables=tables)
            return {rec["name"] for rec in records}

    def fetch_tables(
        self, db_id: str, table_names: Sequence[str]
    ) -> List[GraphTable]:
        if not table_names:
            return []
        query = """
        MATCH (d:Database {name: $db_id})-[:HAS_TABLE]->(t:Table)
        WHERE t.name IN $tables
        OPTIONAL MATCH (t)-[:HAS_COLUMN]->(c:Column)
        RETURN t.name AS table, t.friendly_name AS friendly,
               collect({
                 name: c.name, friendly_name: c.friendly_name, type: c.type,
                 is_primary_key: c.is_primary_key, is_foreign_key: c.is_foreign_key,
                 description: c.description, value_description: c.value_description,
                 sample_values: c.sample_values, ordinal: c.ordinal
               }) AS columns
        """
        tables: List[GraphTable] = []
        with self._driver.session(database=self._database) as session:
            for rec in session.run(query, db_id=db_id, tables=list(table_names)):
                columns = [
                    GraphColumn(
                        name=col["name"],
                        friendly_name=col.get("friendly_name") or "",
                        type=col.get("type") or "",
                        is_primary_key=bool(col.get("is_primary_key")),
                        is_foreign_key=bool(col.get("is_foreign_key")),
                        description=col.get("description") or "",
                        value_description=col.get("value_description") or "",
                        sample_values=col.get("sample_values") or [],
                        ordinal=col.get("ordinal") if col.get("ordinal") is not None else 0,
                    )
                    for col in rec["columns"]
                    if col.get("name") is not None
                ]
                columns.sort(key=lambda c: c.ordinal)
                tables.append(
                    GraphTable(
                        name=rec["table"],
                        friendly_name=rec.get("friendly") or rec["table"],
                        columns=columns,
                    )
                )
        return tables

    def fetch_foreign_keys(
        self, db_id: str, table_names: Sequence[str]
    ) -> List[GraphForeignKey]:
        if not table_names:
            return []
        query = """
        MATCH (d:Database {name: $db_id})-[:HAS_TABLE]->(t1:Table)
              -[:HAS_COLUMN]->(c1:Column)-[:REFERENCES]->(c2:Column)
              <-[:HAS_COLUMN]-(t2:Table)<-[:HAS_TABLE]-(d)
        WHERE t1.name IN $tables AND t2.name IN $tables
        RETURN t1.name AS from_table, c1.name AS from_column,
               t2.name AS to_table, c2.name AS to_column
        """
        fks: List[GraphForeignKey] = []
        with self._driver.session(database=self._database) as session:
            for rec in session.run(query, db_id=db_id, tables=list(table_names)):
                fks.append(
                    GraphForeignKey(
                        from_table=rec["from_table"],
                        from_column=rec["from_column"],
                        to_table=rec["to_table"],
                        to_column=rec["to_column"],
                    )
                )
        return fks

    def close(self) -> None:
        self._driver.close()
