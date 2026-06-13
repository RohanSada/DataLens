import sqlite3

import pytest

from app.core.config import Settings
from app.services.sql_executor import SqlExecutor


@pytest.fixture
def executor():
    return SqlExecutor(Settings())


def test_validate_rejects_non_select(executor):
    with pytest.raises(ValueError, match="Only SELECT"):
        executor.validate("DELETE FROM users")


def test_validate_rejects_multi_statement(executor):
    with pytest.raises(ValueError, match="single SQL"):
        executor.validate("SELECT 1; SELECT 2")


def test_validate_accepts_select(executor):
    executor.validate("SELECT id, name FROM users WHERE id = 1")


def test_execute_returns_rows(executor, tmp_path):
    db_path = tmp_path / "test.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE items (id INTEGER, label TEXT)")
    conn.execute("INSERT INTO items VALUES (1, 'a')")
    conn.commit()

    columns, rows = executor.execute(conn, "SELECT id, label FROM items")
    assert columns == ["id", "label"]
    assert rows == [[1, "a"]]
    conn.close()


def test_execute_enforces_row_cap():
    settings = Settings(max_query_rows=2)
    executor = SqlExecutor(settings)
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (id INTEGER)")
    conn.executemany("INSERT INTO t VALUES (?)", [(i,) for i in range(10)])
    conn.commit()

    _, rows = executor.execute(conn, "SELECT id FROM t")
    assert len(rows) == 2
