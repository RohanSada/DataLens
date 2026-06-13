from __future__ import annotations

from typing import Any

from app.models.requests import ConnectRequest
from app.services.db_connection import DatabaseSession
from app.services.schema_extractor import extract_bird_schema
from app.services.schema_extractor_dialects import (
    extract_mysql_schema,
    extract_postgres_schema,
)


class SchemaService:
    """Extract, store, and render BIRD-format schemas for the LLM."""

    def resolve_db_id(self, request: ConnectRequest) -> str:
        if request.db_id and request.db_id.strip():
            return request.db_id.strip()
        if request.database and request.database.strip():
            return request.database.strip()
        if request.file_id:
            return request.file_id
        return "database"

    def extract_bird_schema(self, session: DatabaseSession, db_id: str) -> dict[str, Any]:
        if session.db_type == "sqlite":
            return extract_bird_schema(session.connection, db_id)
        if session.db_type == "postgres":
            return extract_postgres_schema(session.connection, db_id)
        if session.db_type == "mysql":
            return extract_mysql_schema(session.connection, db_id)
        raise NotImplementedError(
            f"BIRD schema extraction not implemented for {session.db_type}"
        )

    def render_for_llm(self, schema: dict[str, Any]) -> str:
        """Render a stored BIRD schema record as readable text for the model."""
        table_names = schema["table_names_original"]
        lines: list[str] = [f"Database: {schema['db_id']}", ""]

        for table_idx, table_name in enumerate(table_names):
            lines.append(f"Table: {table_name}")
            for col_entry, col_type in zip(
                self._columns_for_table(schema, table_idx),
                self._types_for_table(schema, table_idx),
            ):
                lines.append(f"  - {col_entry[1]} ({col_type})")
            lines.append("")

        if schema.get("foreign_keys"):
            lines.append("Foreign keys:")
            for from_idx, to_idx in schema["foreign_keys"]:
                from_col = schema["column_names_original"][from_idx]
                to_col = schema["column_names_original"][to_idx]
                from_table = table_names[from_col[0]]
                to_table = table_names[to_col[0]]
                lines.append(
                    f"  - {from_table}.{from_col[1]} -> {to_table}.{to_col[1]}"
                )

        return "\n".join(lines).strip()

    def _columns_for_table(
        self, schema: dict[str, Any], table_idx: int
    ) -> list[list[Any]]:
        return [
            entry
            for entry in schema["column_names_original"]
            if entry[0] == table_idx
        ]

    def _types_for_table(self, schema: dict[str, Any], table_idx: int) -> list[str]:
        types: list[str] = []
        for entry in schema["column_names_original"]:
            if entry[0] == -1:
                continue
            if entry[0] == table_idx:
                col_pos = schema["column_names_original"].index(entry) - 1
                types.append(schema["column_types"][col_pos])
        return types
