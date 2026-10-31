"""Prompts for the agent, the selector, and the observations the agent sees.

Nothing here is specific to a database or a benchmark question. The rules about
answer shape describe how execution accuracy grades a query (rows compared as
sets of tuples), which every BIRD-style benchmark shares.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

ANSWER_RULES = """\
How the final query is graded: it is executed, and its rows are compared with the rows of a reference \
query as sets of tuples. Row order and duplicate rows don't matter. Column names don't matter, but the \
number, order and values of the columns do. So:
- Return exactly what the question asks for, one column per requested item, in the order the question \
mentions them. Don't add IDs, names, counts or other helper columns that weren't asked for.
- Follow the external knowledge literally. It defines the terms, value mappings and formulas the reference \
query uses (for example, if it says "percentage = DIVIDE(SUM(x), COUNT(y)) * 100", compute exactly that).
- For "the most / highest / lowest / top ..." return the top row(s) with ORDER BY ... LIMIT n, unless the \
question asks for every tied row.
- Use CAST(... AS REAL) when dividing integers. Don't round or format numbers unless the question asks.
- Text comparisons in SQLite are case-sensitive: use values exactly as stored in the database.
- Keep stored columns separate (for example first and last name), unless the question asks to combine them.
- When ranking by or aggregating a column with NULLs, exclude the NULLs.
- Use only tables and columns that exist. Quote names containing spaces or symbols with backticks."""

AGENT_SYSTEM = f"""\
You are an expert data analyst. You answer questions about a SQLite database by writing one SQLite \
query, and you can query the database before you answer.

Work in turns. In each turn, think briefly, then write exactly one action:

<sql>
SELECT ...
</sql>
Runs a read-only query and shows you the first rows of its result. Use it to inspect values, check \
joins and filters, and test a draft of your final query.

<search>text</search>
Finds text values in the database that equal or resemble "text", and the columns that hold them. Use it \
to get the exact spelling of names, codes and categories from the question.

<answer>
SELECT ...
</answer>
Your final query. It ends the task.

After a <sql> or <search> action, stop and wait for the result; never write results yourself. A good \
process: identify the tables, columns and literal values you need, check uncertain ones, run your draft \
query to see that its result has the right shape and plausible values, then answer.

{ANSWER_RULES}"""

DIRECT_SYSTEM = f"""\
You are an expert data analyst. You answer questions about a SQLite database by writing one SQLite query.

Think through the tables, joins, filters and aggregation you need, then give the final query as:

<answer>
SELECT ...
</answer>

{ANSWER_RULES}"""

FORMAT_REMINDER = (
    "Reply with exactly one action: <sql>...</sql>, <search>...</search> or <answer>...</answer>."
)
LAST_TURN = "This is your last turn: give your final query now as <answer>...</answer>."
TURNS_LEFT = "({n} turns left.)"


def schema_block(schema_text: str) -> str:
    return f"Database engine: SQLite\n\nDatabase schema:\n{schema_text}\n\n"


def question_block(question: str, evidence: str, hints: Sequence[str]) -> str:
    parts = []
    if hints:
        parts.append(
            "Values from the question found in the database (exact matches, may be incomplete):\n"
            + "\n".join(f"- {h}" for h in hints)
        )
    parts.append(f"External knowledge: {evidence.strip() or 'None'}")
    parts.append(f"Question: {question.strip()}")
    return "\n\n".join(parts)


def _cell(value: Any, limit: int = 60) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def format_rows(
    columns: Sequence[str] | None,
    rows: Sequence[Sequence[Any]],
    *,
    total: int | None = None,
    truncated: bool = False,
    max_rows: int = 10,
) -> str:
    """A compact text table: column header, then up to ``max_rows`` rows."""
    total = len(rows) if total is None else total
    if total == 0:
        header = " | ".join(columns or [])
        return f"columns: {header}\n(0 rows)"
    count = f"more than {total}" if truncated else str(total)
    shown = rows[:max_rows]
    lines = [f"columns: {' | '.join(columns or [])}", f"rows ({count} total, showing {len(shown)}):"]
    lines.extend(" | ".join(_cell(v) for v in row) for row in shown)
    return "\n".join(lines)


SELECTOR_SYSTEM = """\
You are an expert data analyst reviewing candidate SQLite queries written for a question. The candidates \
return different results, and at most one of them matches the reference answer. Judge which candidate \
answers the question exactly as asked: the right tables and join, the right filters and literal values, \
the formulas in the external knowledge, and exactly the requested columns (no missing or extra ones)."""


def selector_prompt(question: str, evidence: str, candidates: Sequence[tuple[str, str, str]]) -> str:
    """``candidates`` are ``(label, sql, result_text)``."""
    lines = [f"External knowledge: {evidence.strip() or 'None'}", f"Question: {question.strip()}", ""]
    for label, sql, result in candidates:
        lines += [f"Candidate {label}:", "```sql", sql.strip(), "```", "Result:", result, ""]
    letters = ", ".join(label for label, _, _ in candidates)
    lines.append(
        "Compare the candidates against the question and the external knowledge. Point out the decisive "
        f"differences briefly, then name the best candidate ({letters}) on the last line as "
        "<choice>X</choice>."
    )
    return "\n".join(lines)
