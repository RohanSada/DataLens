"""Prompt format aligned with Training/BIRD_SQL/prompt_template.py."""

SYSTEM_INSTRUCTION = (
    "You are an expert data analyst who writes SQLite queries. "
    "Using the database schema below, write one valid SQLite query that "
    "answers the question. Output only the SQL query, nothing else."
)

SCHEMA_HEADER = "### Database Schema"
QUESTION_HEADER = "### Question"
SQL_HEADER = "### SQL"


def build_prompt(schema_context: str, question: str) -> str:
    parts = [
        SYSTEM_INSTRUCTION,
        "",
        SCHEMA_HEADER,
        schema_context.strip(),
        "",
        QUESTION_HEADER,
        question.strip(),
        "",
        SQL_HEADER,
        "",
    ]
    return "\n".join(parts)
