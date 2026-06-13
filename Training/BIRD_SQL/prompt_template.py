"""Render the final ``{prompt, completion}`` pair for one example.

The prompt is a plain instruction format (not a chat template) so the resulting
JSONL is framework-agnostic: it works with TRL's ``SFTTrainer`` on a
``prompt``/``completion`` dataset, with completion-only loss masking, or with a
manual collator. The target ``completion`` is the gold SQL only.
"""
from __future__ import annotations

SYSTEM_INSTRUCTION = (
    "You are an expert data analyst who writes SQLite queries. "
    "Using the database schema below, write one valid SQLite query that "
    "answers the question. Output only the SQL query, nothing else."
)

SCHEMA_HEADER = "### Database Schema"
QUESTION_HEADER = "### Question"
HINT_HEADER = "### Hint"
SQL_HEADER = "### SQL"


def build_prompt(schema_context: str, question: str, evidence: str = "") -> str:
    """Assemble the prompt string fed to the model.

    The prompt ends with the ``### SQL`` header followed by a newline, so the
    model's continuation is exactly the query.
    """
    parts = [
        SYSTEM_INSTRUCTION,
        "",
        SCHEMA_HEADER,
        schema_context.strip(),
        "",
        QUESTION_HEADER,
        question.strip(),
    ]
    if evidence and evidence.strip():
        parts += ["", HINT_HEADER, evidence.strip()]
    parts += ["", SQL_HEADER, ""]
    return "\n".join(parts)


def build_completion(sql: str) -> str:
    """Normalize the gold SQL into the target completion (single line)."""
    return " ".join(sql.strip().split())


def build_example(
    schema_context: str,
    question: str,
    evidence: str,
    sql: str,
) -> dict:
    """Return the ``{prompt, completion}`` record body (without metadata)."""
    return {
        "prompt": build_prompt(schema_context, question, evidence),
        "completion": build_completion(sql),
    }
