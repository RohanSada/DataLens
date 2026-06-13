"""Render retrieved schema into a compact, prompt-ready string.

Output is CREATE TABLE-style DDL with inline comments for descriptions and value
hints, followed by an explicit list of foreign-key join paths. The format is
both LLM-friendly and close to real SQL the model already understands.
"""
from __future__ import annotations

from typing import List, Sequence

from graph_store import GraphForeignKey, GraphTable


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) - avoids a tokenizer dependency."""
    return (len(text) + 3) // 4


def _column_line(column, highlight: bool, is_last: bool) -> str:
    definition = f"  {column.name} {column.type or 'TEXT'}".rstrip()
    if column.is_primary_key:
        definition += " PRIMARY KEY"
    if not is_last:
        definition += ","

    comment_bits: List[str] = []
    if highlight:
        comment_bits.append("[relevant]")
    if column.description:
        comment_bits.append(column.description)
    if column.value_description:
        comment_bits.append(f"values: {column.value_description}")
    elif column.sample_values:
        comment_bits.append("e.g. " + ", ".join(column.sample_values))

    if comment_bits:
        comment = "; ".join(b.replace("\n", " ").strip() for b in comment_bits)
        return f"{definition}  -- {comment}"
    return definition


def render_table(table: GraphTable, highlight_columns: Sequence[str]) -> str:
    highlight = {c.lower() for c in highlight_columns}
    lines = [f"CREATE TABLE {table.name} ("]
    last_idx = len(table.columns) - 1
    for idx, column in enumerate(table.columns):
        lines.append(
            _column_line(column, column.name.lower() in highlight, idx == last_idx)
        )
    lines.append(");")
    return "\n".join(lines)


def render_schema(
    tables: Sequence[GraphTable],
    foreign_keys: Sequence[GraphForeignKey],
    highlight_columns: Sequence[str] = (),
    token_budget: int | None = None,
) -> str:
    """Render tables + FK relationships, trimming to a token budget if given."""
    blocks: List[str] = []
    running = 0
    rendered_tables = set()

    for table in tables:
        block = render_table(table, highlight_columns)
        cost = estimate_tokens(block)
        if token_budget is not None and running + cost > token_budget and blocks:
            break
        blocks.append(block)
        rendered_tables.add(table.name)
        running += cost

    fk_lines = [
        f"-- {fk.from_table}.{fk.from_column} -> {fk.to_table}.{fk.to_column}"
        for fk in foreign_keys
        if fk.from_table in rendered_tables and fk.to_table in rendered_tables
    ]
    if fk_lines:
        blocks.append("-- Foreign keys (join paths):\n" + "\n".join(fk_lines))

    return "\n\n".join(blocks)
