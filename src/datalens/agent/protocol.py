"""Parsing one agent turn into an action.

The agent writes plain text with one tagged action per turn (``<sql>``,
``<search>`` or ``<answer>``). Plain tags work with any chat model and any
serving stack, unlike native tool calling, whose format differs per model
family. Generation stops at the closing tag, so the model can't invent the
result of its own query.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from datalens.sql.parse import extract_sql

ActionKind = Literal["sql", "search", "answer", "none"]
TAGS: tuple[str, ...] = ("sql", "search", "answer")
STOP_SEQUENCES: tuple[str, ...] = tuple(f"</{t}>" for t in TAGS)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_OPEN_RE = re.compile(r"<(sql|search|answer)>", re.IGNORECASE)


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    body: str = ""

    @property
    def sql(self) -> str | None:
        """The query in a ``sql`` or ``answer`` action.

        Text that doesn't look like a SELECT is passed through as is, so the
        database's error message (or the read-only check) tells the model what's wrong.
        """
        if self.kind not in ("sql", "answer"):
            return None
        return extract_sql(self.body) or self.body.strip().rstrip(";").strip() or None


def visible_text(text: str) -> str:
    """Model output without reasoning blocks; an unclosed ``<think>`` hides the rest."""
    body = _THINK_RE.sub("", text)
    lowered = body.lower()
    if "<think>" in lowered:
        body = body[: lowered.index("<think>")]
    return body


def parse_action(text: str, stop_sequence: str | None = None) -> Action:
    """The first action in a turn.

    ``stop_sequence`` is the closing tag generation stopped at, if any (it isn't
    part of ``text``). Without tags, a fenced SQL block is taken as the answer, so
    a model that forgets the format still gets graded on its query.
    """
    body = visible_text(text)
    match = _OPEN_RE.search(body)
    if match:
        kind = match.group(1).lower()
        rest = body[match.end() :]
        close = rest.lower().find(f"</{kind}>")
        content = rest[:close] if close >= 0 else rest
        content = content.strip()
        if kind == "search":
            content = content.strip().strip("\"'`").strip()
            return Action("search", content) if content else Action("none")
        if not content.strip("`").strip():
            return Action("none")
        return Action(kind, content)  # type: ignore[arg-type]
    if extract_sql(body) is not None and "```" in body:
        return Action("answer", body[body.find("```") :])
    return Action("none")


def close_turn(text: str, stop_sequence: str | None) -> str:
    """The assistant message as it goes back into the history (closing tag restored)."""
    if stop_sequence and not text.rstrip().endswith(stop_sequence):
        return text.rstrip() + "\n" + stop_sequence
    return text
