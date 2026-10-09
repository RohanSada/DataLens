"""The single prompt every model is evaluated with.

Small fine-tuned models and large zero-shot models see byte-identical text, so a
difference in accuracy can't come from prompt engineering. The prompt is built
in parts so backends that support prompt caching (Anthropic) can cache the
schema prefix that repeats across questions on the same database.
"""

from __future__ import annotations

from dataclasses import dataclass

from datalens.data.benchmarks import Example
from datalens.data.schema import load_schema

SYSTEM_PROMPT = (
    "You are an expert data analyst. You translate questions into correct, "
    "efficient SQLite queries over the database you are given."
)

_REASONING_INSTRUCTIONS = (
    "Instructions:\n"
    "- Use only tables and columns that exist in the schema.\n"
    "- Use the external knowledge when it defines a term or a calculation.\n"
    "- First reason about the tables, joins, filters and aggregation you need "
    "inside <think> </think> tags.\n"
    "- Then give the final SQLite query in a single ```sql code block."
)

_DIRECT_INSTRUCTIONS = (
    "Instructions:\n"
    "- Use only tables and columns that exist in the schema.\n"
    "- Use the external knowledge when it defines a term or a calculation.\n"
    "- Reply with the SQLite query only, in a single ```sql code block."
)


@dataclass(frozen=True)
class PromptParts:
    """A prompt split into a cacheable schema prefix and a per-question suffix."""

    system: str
    schema_block: str
    question_block: str

    @property
    def user(self) -> str:
        return self.schema_block + self.question_block

    def messages(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
        ]


def build_prompt(
    schema_text: str,
    question: str,
    evidence: str = "",
    *,
    reasoning: bool = True,
) -> PromptParts:
    """Assemble the prompt for one question.

    ``reasoning=True`` asks for ``<think>`` reasoning before the query, which is
    the format GRPO trains; ``reasoning=False`` asks for the query alone, which is
    the format SFT on gold SQL trains.
    """
    schema_block = f"Database engine: SQLite\n\nDatabase schema:\n{schema_text}\n\n"
    knowledge = evidence.strip() or "None"
    instructions = _REASONING_INSTRUCTIONS if reasoning else _DIRECT_INSTRUCTIONS
    question_block = f"External knowledge: {knowledge}\n\nQuestion: {question.strip()}\n\n{instructions}"
    return PromptParts(system=SYSTEM_PROMPT, schema_block=schema_block, question_block=question_block)


def format_answer(sql: str, reasoning: str | None = None) -> str:
    """Target completion text for supervised fine-tuning."""
    answer = f"```sql\n{sql.strip()}\n```"
    if reasoning:
        return f"<think>\n{reasoning.strip()}\n</think>\n\n{answer}"
    return answer


@dataclass(frozen=True)
class PromptConfig:
    """How schemas are rendered and which answer format is requested."""

    reasoning: bool = True
    num_examples: int = 3
    descriptions: bool = False
    schema_cache_dir: str | None = None


def prompt_for_example(example: Example, config: PromptConfig) -> PromptParts:
    """Build the prompt for a benchmark example under ``config``."""
    schema = load_schema(example.db_path, num_examples=config.num_examples, cache_dir=config.schema_cache_dir)
    schema_text = schema.render(examples=config.num_examples > 0, descriptions=config.descriptions)
    return build_prompt(schema_text, example.question, example.evidence, reasoning=config.reasoning)
