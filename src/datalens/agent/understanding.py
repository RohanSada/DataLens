"""Step 1, task understanding: literals, question skeleton, matched values, examples.

The task-understanding model returns JSON with the database literals of the
question and its skeleton (paper Fig. 9-10). Small models do not always return
valid JSON, so parsing is lenient, and anything missing falls back to
heuristics: n-grams of the question still drive value matching, and
:func:`~datalens.agent.examples.heuristic_skeleton` supplies a skeleton.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from datalens.agent.examples import Demo
from datalens.agent.values import ValueMatch

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


@dataclass
class Understanding:
    """Everything step 1 found out about one question."""

    literals: list[str] = field(default_factory=list)
    skeleton: str = ""
    matches: list[ValueMatch] = field(default_factory=list)
    demos: list[Demo] = field(default_factory=list)  # few-shot examples for the ICL generator
    source: str = "llm"  # "llm", "heuristic" (model output unusable) or "disabled"

    def matched(self) -> dict[str, list[str]]:
        """``{"table.column": [values]}`` for schema rendering."""
        grouped: dict[str, list[str]] = {}
        for m in self.matches:
            values = grouped.setdefault(m.key, [])
            if m.value not in values:
                values.append(m.value)
        return grouped

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Understanding:
        return cls(
            literals=list(data.get("literals") or []),
            skeleton=data.get("skeleton") or "",
            matches=[ValueMatch(**m) for m in data.get("matches") or []],
            demos=[Demo(**d) for d in data.get("demos") or []],
            source=data.get("source") or "llm",
        )


def _json_objects(text: str) -> list[str]:
    """Candidate JSON object strings in ``text``: fenced blocks, then the outermost braces."""
    body = _THINK_RE.sub("", text or "")
    found = [m.strip() for m in _FENCE_RE.findall(body)]
    start, end = body.find("{"), body.rfind("}")
    if start != -1 and end > start:
        found.append(body[start : end + 1])
    return found


def parse_understanding(text: str) -> tuple[list[str], str | None]:
    """``(literals, skeleton)`` from the model's JSON; ``([], None)`` if unusable."""
    for candidate in _json_objects(text):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        raw_literals = data.get("database_literals") or []
        if isinstance(raw_literals, str):
            raw_literals = [raw_literals]
        literals = []
        if isinstance(raw_literals, list):
            for item in raw_literals:
                if isinstance(item, (str, int, float)) and str(item).strip():
                    literals.append(str(item).strip())
        skeleton = data.get("question_skeleton")
        skeleton = skeleton.strip() if isinstance(skeleton, str) and skeleton.strip() else None
        return list(dict.fromkeys(literals)), skeleton
    # Not JSON at all: salvage a "Skeleton: ..." line if the model wrote one.
    match = re.search(r"skeleton\W*:\s*(.+)", text or "", re.IGNORECASE)
    return [], (match.group(1).strip().strip('"') if match else None)
