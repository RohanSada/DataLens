from __future__ import annotations

import concurrent.futures
from typing import Any

import sqlglot
from sqlglot import exp

from app.core.config import Settings


class SqlExecutor:
    """Validate and execute generated SQL safely."""

    DIALECT_MAP = {
        "sqlite": "sqlite",
        "postgres": "postgres",
        "mysql": "mysql",
    }

    def __init__(self, settings: Settings):
        self.settings = settings

    def validate(self, sql: str, dialect: str = "sqlite") -> None:
        normalized = sql.strip()
        if not normalized:
            raise ValueError("Empty SQL is not allowed")

        read_dialect = self.DIALECT_MAP.get(dialect, "sqlite")
        try:
            statements = sqlglot.parse(normalized, read=read_dialect)
        except Exception as exc:
            raise ValueError(f"Invalid SQL: {exc}") from exc

        if len(statements) != 1:
            raise ValueError("Only a single SQL statement is allowed")

        statement = statements[0]
        if not isinstance(statement, exp.Select):
            raise ValueError("Only SELECT queries are allowed")

        for node in statement.walk():
            if isinstance(
                node,
                (
                    exp.Insert,
                    exp.Update,
                    exp.Delete,
                    exp.Drop,
                    exp.Alter,
                    exp.Create,
                    exp.TruncateTable,
                    exp.Command,
                ),
            ):
                raise ValueError("Unsafe SQL detected")

    def execute(
        self,
        connection: Any,
        sql: str,
        dialect: str = "sqlite",
    ) -> tuple[list[str], list[list[Any]]]:
        self.validate(sql, dialect=dialect)
        return self._execute_with_timeout(connection, sql, dialect)

    def _execute_with_timeout(
        self,
        connection: Any,
        sql: str,
        dialect: str,
    ) -> tuple[list[str], list[list[Any]]]:
        if dialect == "sqlite":
            return self._run_query(connection, sql, dialect)

        timeout = self.settings.query_timeout_seconds
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._run_query, connection, sql, dialect)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError as exc:
                raise ValueError(
                    f"Query exceeded timeout of {timeout} seconds"
                ) from exc

    def _run_query(
        self,
        connection: Any,
        sql: str,
        dialect: str,
    ) -> tuple[list[str], list[list[Any]]]:
        if dialect == "postgres":
            with connection.cursor() as cursor:
                cursor.execute(f"SET statement_timeout = {self.settings.query_timeout_seconds * 1000}")
                cursor.execute(sql)
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchmany(self.settings.max_query_rows)
                return columns, [list(row) for row in rows]

        if dialect == "mysql":
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SET SESSION MAX_EXECUTION_TIME={self.settings.query_timeout_seconds * 1000}"
                )
                cursor.execute(sql)
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchmany(self.settings.max_query_rows)
                return columns, [list(row) for row in rows]

        cursor = connection.cursor()
        cursor.execute(f"PRAGMA busy_timeout = {self.settings.query_timeout_seconds * 1000}")
        cursor.execute(sql)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchmany(self.settings.max_query_rows)
        return columns, [list(row) for row in rows]
