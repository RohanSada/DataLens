"""Choosing one answer from several attempts.

Attempts are grouped by the rows their answers return. A group holding a clear
majority of attempts wins outright (execution-based self-consistency). Otherwise
the model judges the largest distinct results side by side, a few times with
the candidates in different orders, and its votes are combined with the group
sizes. Position bias is the main failure of LLM judges, which the shuffling
averages out.
"""

from __future__ import annotations

import random
import re
import zlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from datalens.agent.chat import ChatSampling, Conversation
from datalens.agent.config import AgentSpec
from datalens.agent.prompts import SELECTOR_SYSTEM, selector_prompt
from datalens.agent.rollout import Attempt, Database

_CHOICE_RE = re.compile(r"<choice>\s*\(?([A-Z])\)?\s*</choice>", re.IGNORECASE)
LABELS = "ABCDEFGH"


@dataclass
class Group:
    key: str
    members: list[int]  # attempt indices, in order
    sql: str
    rows: int
    empty: bool

    @property
    def votes(self) -> int:
        return len(self.members)


@dataclass
class Selection:
    sql: str | None
    method: str  # single | consensus | selector | vote | none
    groups: list[Group] = field(default_factory=list)
    shown: list[int] = field(default_factory=list)  # group indices shown to the selector
    scores: list[float] = field(default_factory=list)
    judgements: list[dict[str, Any]] = field(default_factory=list)
    chosen_group: int | None = None


def group_attempts(attempts: Sequence[Attempt]) -> list[Group]:
    """Groups of attempts with identical results, largest first.

    Ties go to non-empty results, then to the group whose first member came
    earliest (the greedy attempt, when there is one).
    """
    groups: dict[str, Group] = {}
    for a in attempts:
        key = a.key
        if key is None or a.answer_sql is None or a.answer_result is None:
            continue
        rows = len(a.answer_result.rows or [])
        if key not in groups:
            groups[key] = Group(key=key, members=[], sql=a.answer_sql, rows=rows, empty=rows == 0)
        groups[key].members.append(a.index)
    return sorted(groups.values(), key=lambda g: (-g.votes, g.empty, g.members[0]))


def preliminary(attempts: Sequence[Attempt], spec: AgentSpec) -> Selection:
    """Decide without the selector where possible; ``method == "selector"`` means it is needed."""
    groups = group_attempts(attempts)
    if not groups:
        failed = next((a.answer_sql for a in attempts if a.answer_sql), None)
        return Selection(sql=failed, method="none")
    top = groups[0]
    if len(groups) == 1:
        return Selection(
            sql=top.sql, method="single" if len(attempts) == 1 else "consensus", groups=groups, chosen_group=0
        )
    share = top.votes / len(attempts)
    if spec.selection == "vote" or share >= spec.consensus:
        return Selection(
            sql=top.sql,
            method="consensus" if share >= spec.consensus else "vote",
            groups=groups,
            chosen_group=0,
        )
    shown = list(range(min(spec.selector_candidates, len(groups))))
    return Selection(sql=top.sql, method="selector", groups=groups, shown=shown, chosen_group=0)


def _orders(question_id: str, n: int, samples: int, seed: int) -> list[list[int]]:
    """Candidate orders for each judgement: the first as ranked, the rest shuffled."""
    rng = random.Random(zlib.crc32(f"{seed}:{question_id}".encode()))
    orders = [list(range(n))]
    for _ in range(samples - 1):
        order = list(range(n))
        rng.shuffle(order)
        orders.append(order)
    return orders


def selector_requests(
    question_id: str,
    question: str,
    evidence: str,
    schema_text: str,
    selection: Selection,
    db: Database,
    attempts: Sequence[Attempt],
    spec: AgentSpec,
) -> list[tuple[list[int], Conversation, ChatSampling]]:
    """Conversations asking the model to judge the shown groups (one per judgement)."""
    by_index = {a.index: a for a in attempts}
    results = []
    for gi in selection.shown:
        group = selection.groups[gi]
        result = by_index[group.members[0]].answer_result
        results.append(db.describe(result) if result is not None else "(no result)")
    requests = []
    for j, order in enumerate(_orders(question_id, len(selection.shown), spec.selector_samples, spec.seed)):
        candidates = [
            (LABELS[pos], selection.groups[selection.shown[k]].sql, results[k]) for pos, k in enumerate(order)
        ]
        prompt = selector_prompt(question, evidence, candidates)
        conversation: Conversation = [
            {"role": "system", "content": SELECTOR_SYSTEM},
            {"role": "user", "content": schema_text + prompt, "parts": [schema_text, prompt]},
        ]
        sampling = ChatSampling(
            temperature=0.0 if j == 0 else spec.selector_temperature,
            top_p=None if j == 0 else spec.top_p,
            max_tokens=spec.selector_max_tokens,
            seed=zlib.crc32(f"sel:{spec.seed}:{question_id}:{j}".encode()),
        )
        requests.append((order, conversation, sampling))
    return requests


def parse_choice(text: str, n: int) -> int | None:
    """Position (0-based) of the chosen candidate, from the last ``<choice>`` tag."""
    matches = _CHOICE_RE.findall(text or "")
    if not matches:
        return None
    pos = LABELS.find(matches[-1].upper())
    return pos if 0 <= pos < n else None


def finish_selection(
    selection: Selection, judgements: Sequence[tuple[list[int], str]], total_attempts: int, spec: AgentSpec
) -> Selection:
    """Combine selector votes with group sizes and pick the winner."""
    n = len(selection.shown)
    votes = [0.0] * n
    for order, text in judgements:
        pos = parse_choice(text, n)
        selection.judgements.append({"order": order, "choice": None if pos is None else order[pos]})
        if pos is not None:
            votes[order[pos]] += 1
    shares = [selection.groups[g].votes / max(total_attempts, 1) for g in selection.shown]
    scores = [v + spec.vote_weight * s for v, s in zip(votes, shares, strict=True)]
    # Ties: larger group, then the earlier-ranked group.
    best = max(range(n), key=lambda k: (scores[k], shares[k], -k))
    selection.scores = scores
    selection.chosen_group = selection.shown[best]
    selection.sql = selection.groups[selection.chosen_group].sql
    return selection
