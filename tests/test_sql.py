from __future__ import annotations

import sqlite3

import pytest

from datalens.sql.compare import exec_match, result_key, soft_f1, ves_reward
from datalens.sql.executor import ErrorKind, execute
from datalens.sql.parse import extract_sql, is_read_only, schema_refs

# A truncated model answer from a BIRD dev run: the unclosed quote stops sqlglot's tokenizer.
_UNCLOSED_QUOTE = "SELECT f.School FROM frpm AS f ORDER BY f.`FRPM Count (K-12)` DESC LIMIT '"


class TestExecute:
    def test_returns_rows_and_columns(self, shop_db):
        result = execute(shop_db, "SELECT name FROM customers ORDER BY id")
        assert result.ok
        assert result.columns == ["name"]
        assert result.rows == [("Ada",), ("Grace",), ("Linus",)]

    @pytest.mark.parametrize(
        ("sql", "kind"),
        [
            ("SELEC name FROM customers", ErrorKind.SYNTAX),
            ("SELECT name FROM clients", ErrorKind.SCHEMA),
            ("SELECT nme FROM customers", ErrorKind.SCHEMA),
            ("SELECT abs('a', 'b')", ErrorKind.RUNTIME),
        ],
    )
    def test_classifies_errors(self, shop_db, sql, kind):
        result = execute(shop_db, sql)
        assert not result.ok
        assert result.error_kind is kind

    def test_times_out_runaway_queries(self, shop_db):
        sql = "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n) SELECT max(x) FROM n"
        result = execute(shop_db, sql, timeout_s=0.2)
        assert result.error_kind is ErrorKind.TIMEOUT
        assert result.elapsed_s < 5

    def test_database_is_read_only(self, shop_db):
        result = execute(shop_db, "DELETE FROM customers")
        assert not result.ok
        assert sqlite3.connect(shop_db).execute("SELECT COUNT(*) FROM customers").fetchone() == (3,)

    def test_row_cap_marks_truncation(self, shop_db):
        result = execute(shop_db, "SELECT * FROM orders", max_rows=2)
        assert result.ok and result.truncated and len(result.rows) == 2

    def test_missing_database(self, tmp_path):
        result = execute(tmp_path / "nope.sqlite", "SELECT 1")
        assert not result.ok and "not found" in result.error


class TestCompare:
    def test_exec_match_ignores_order_and_duplicates(self):
        assert exec_match([(1,), (2,), (2,)], [(2,), (1,)])
        assert not exec_match([(1,)], [(1,), (2,)])

    def test_exec_match_is_column_order_sensitive(self):
        assert not exec_match([(1, "a")], [("a", 1)])

    def test_result_key_matches_exec_match_semantics(self):
        assert result_key([(1,), (2,)]) == result_key([(2,), (1,), (1,)])
        assert result_key([(1.0,)]) == result_key([(1,)])
        assert result_key([(1,)]) != result_key([(2,)])

    def test_soft_f1(self):
        assert soft_f1([], []) == 1.0
        assert soft_f1([(1, 2)], [(1, 2)]) == 1.0
        assert soft_f1([(9,)], [(1,)]) == 0.0
        # one of two cells right in the single row pair -> precision = recall = 0.5
        assert soft_f1([(1, 9)], [(1, 2)]) == pytest.approx(0.5)
        # an extra predicted row costs precision only
        assert soft_f1([(1,), (5,)], [(1,)]) == pytest.approx(2 * 0.5 * 1 / 1.5)

    @pytest.mark.parametrize(
        ("ratio", "reward"), [(0, 0), (3, 1.25), (1.5, 1), (0.6, 0.75), (0.3, 0.5), (0.1, 0.25)]
    )
    def test_ves_buckets(self, ratio, reward):
        assert ves_reward(ratio) == reward


class TestExtractSql:
    def test_takes_last_fenced_block_after_reasoning(self):
        text = (
            "<think>maybe ```sql SELECT 1``` no</think>\n"
            "Draft:\n```sql\nSELECT 2\n```\nFinal:\n```sql\nSELECT 3;\n```"
        )
        assert extract_sql(text) == "SELECT 3"

    @pytest.mark.parametrize(
        "text",
        [
            "```\nSELECT 1\n```",
            "```SQL\r\nSELECT 1\r\n```",
            "```postgresql\nSELECT 1\n```",
            "```sql SELECT 1```",
            "```SELECT 1```",
        ],
    )
    def test_fence_variants(self, text):
        assert extract_sql(text) == "SELECT 1"

    def test_plain_fence_and_bare_sql(self):
        assert extract_sql("```\nSELECT a FROM t\n```") == "SELECT a FROM t"
        assert extract_sql("Here you go:\nSELECT a FROM t WHERE b = 1;") == "SELECT a FROM t WHERE b = 1"

    def test_keeps_line_leading_subqueries(self):
        text = "SELECT a FROM t\nWHERE b IN (\nSELECT b FROM u)"
        assert extract_sql(text) == text

    def test_unfinished_reasoning_has_no_answer(self):
        assert extract_sql("<think>I should\nSELECT name FROM") is None

    def test_none_when_no_sql(self):
        assert extract_sql("I cannot answer that.") is None
        assert extract_sql("") is None

    def test_first_statement_only(self):
        assert extract_sql("```sql\nSELECT 1; DROP TABLE t;\n```") == "SELECT 1"


class TestParse:
    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT 1",
            "WITH x AS (SELECT 1 AS a) SELECT a FROM x",
            "SELECT a FROM t UNION SELECT b FROM u",
        ],
    )
    def test_read_only_accepts_queries(self, sql):
        assert is_read_only(sql)

    @pytest.mark.parametrize(
        "sql",
        [
            "DELETE FROM t",
            "DROP TABLE t",
            "INSERT INTO t VALUES (1)",
            "UPDATE t SET a = 1",
            "SELECT 1; DROP TABLE t",
            "PRAGMA writable_schema = 1",
            "ATTACH DATABASE 'x.db' AS x",
            "not sql at all (",
            _UNCLOSED_QUOTE,
        ],
    )
    def test_read_only_rejects_writes(self, sql):
        assert not is_read_only(sql)

    def test_schema_refs_return_none_for_sql_that_wont_tokenize(self):
        assert schema_refs(_UNCLOSED_QUOTE) is None

    def test_schema_refs_resolve_aliases(self):
        refs = schema_refs(
            "SELECT T1.name, COUNT(*) AS n FROM customers AS T1 JOIN orders AS T2 "
            "ON T1.id = T2.customer_id GROUP BY T1.name ORDER BY n DESC"
        )
        assert refs is not None
        assert refs.tables == {"customers", "orders"}
        assert refs.columns == {"name", "id", "customer_id"}

    def test_schema_refs_skip_ctes(self):
        refs = schema_refs("WITH x AS (SELECT a FROM t) SELECT a FROM x")
        assert refs is not None and refs.tables == {"t"}
