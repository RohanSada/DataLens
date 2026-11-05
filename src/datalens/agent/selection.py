"""Candidates, result groups and the pairwise round-robin tournament (paper section 3.4).

Selection happens in two steps. Candidates that return the same result are
consolidated into one group, represented by one query. The groups then play a
round-robin: for every pair the selector picks the better query, the winner
scores a point, and the group with the most points is the answer. Ties break
toward the group with more votes, which is what self-consistency would pick.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from itertools import combinations
from typing import Any

from datalens.agent.execution import Outcome
from datalens.inference.voting import group_candidates


@dataclass
class Candidate:
    """One SQL query in a question's pool, with where it came from and what it returned."""

    generator: str  # "reasoning", "icl", "fixer" or "revisor"
    source: str  # finer label, e.g. "icl:cot@0.6#0"
    sql: str | None
    outcome: Outcome
    root: str = ""  # the generator ("reasoning" or "icl") of the query this one descends from
    parent: int | None = None  # pool index of the query a fixer or reviser rewrote
    finish_reason: str | None = None
    output_tokens: int | None = None
    text: str | None = None  # raw model output

    def __post_init__(self) -> None:
        if not self.root:
            self.root = self.generator

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["outcome"] = self.outcome.to_json()
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Candidate:
        data = dict(data)
        data["outcome"] = Outcome.from_json(data["outcome"])
        return cls(**data)


@dataclass(frozen=True)
class ResultGroup:
    """Candidates (pool indices) that returned the same result set."""

    key: str
    members: tuple[int, ...]
    empty: bool

    @property
    def votes(self) -> int:
        return len(self.members)

    @property
    def representative(self) -> int:
        return self.members[0]


def group_by_result(pool: Sequence[Candidate], allowed: set[int] | None = None) -> list[ResultGroup]:
    """Executable candidates grouped by result, most votes first (as self-consistency ranks them).

    ``allowed`` restricts the pool to some candidate indices (used by ablations).
    """
    keys = [
        c.outcome.key if c.outcome.ok and (allowed is None or i in allowed) else None
        for i, c in enumerate(pool)
    ]
    empty = [c.outcome.empty for c in pool]
    return [
        ResultGroup(key=str(g.key), members=g.members, empty=g.empty_result)
        for g in group_candidates(keys, empty)
    ]


@dataclass(frozen=True)
class Match:
    """One pairwise comparison: group ``first`` is shown as candidate 1."""

    first: str
    second: str


def _coin(*parts: str) -> bool:
    return hashlib.sha1("\x1f".join(parts).encode("utf-8")).digest()[0] % 2 == 1


def schedule(
    groups: Sequence[ResultGroup],
    *,
    max_candidates: int | None = None,
    both_orders: bool = False,
    seed: str = "",
) -> list[Match]:
    """All pairs among the top groups, each shown in a pseudo-random order.

    Randomising which query is candidate 1 keeps a position bias in the selector
    from favouring one generator; ``both_orders`` judges every pair both ways.
    """
    entrants = list(groups[:max_candidates] if max_candidates else groups)
    matches = []
    for a, b in combinations(entrants, 2):
        if both_orders:
            matches += [Match(a.key, b.key), Match(b.key, a.key)]
        elif _coin(seed, a.key, b.key):
            matches.append(Match(b.key, a.key))
        else:
            matches.append(Match(a.key, b.key))
    return matches


@dataclass
class Selection:
    """The outcome of selection for one question."""

    method: str
    chosen: int | None  # pool index of the answer
    vote: int | None  # what self-consistency would pick
    groups: list[ResultGroup] = field(default_factory=list)
    matches: list[Match] = field(default_factory=list)
    winners: list[int | None] = field(default_factory=list)  # 1, 2 or None (unparseable)
    texts: list[str | None] = field(default_factory=list)  # the selector's raw judgments
    wins: dict[str, float] = field(default_factory=dict)
    note: str | None = None

    def _texts(self) -> list[str | None]:
        return self.texts if len(self.texts) == len(self.matches) else [None] * len(self.matches)

    def to_json(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "chosen": self.chosen,
            "vote": self.vote,
            "groups": [asdict(g) for g in self.groups],
            "matches": [
                {"first": m.first, "second": m.second, "winner": w, "text": t}
                for m, w, t in zip(self.matches, self.winners, self._texts(), strict=True)
            ],
            "wins": self.wins,
            "note": self.note,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Selection:
        groups = [
            ResultGroup(key=g["key"], members=tuple(g["members"]), empty=g["empty"]) for g in data["groups"]
        ]
        matches = [Match(m["first"], m["second"]) for m in data["matches"]]
        return cls(
            method=data["method"],
            chosen=data["chosen"],
            vote=data["vote"],
            groups=groups,
            matches=matches,
            winners=[m["winner"] for m in data["matches"]],
            texts=[m.get("text") for m in data["matches"]],
            wins=data.get("wins") or {},
            note=data.get("note"),
        )


def tally(matches: Sequence[Match], winners: Sequence[int | None]) -> dict[str, float]:
    """Points per group key; an unparseable judgment gives no points."""
    wins: dict[str, float] = {}
    for match, winner in zip(matches, winners, strict=True):
        wins.setdefault(match.first, 0.0)
        wins.setdefault(match.second, 0.0)
        if winner == 1:
            wins[match.first] += 1
        elif winner == 2:
            wins[match.second] += 1
    return wins


def choose(groups: Sequence[ResultGroup], wins: Mapping[str, float]) -> ResultGroup | None:
    """The group with most wins; ties go to more votes, then non-empty, then the earliest query."""
    if not groups:
        return None
    return min(groups, key=lambda g: (-wins.get(g.key, 0.0), -g.votes, g.empty, g.representative))


def restricted_choice(
    pool: Sequence[Candidate], selection: Selection, allowed: set[int]
) -> tuple[int | None, int | None]:
    """(tournament pick, vote pick) using only candidates in ``allowed``.

    Reuses the judgments already made between the groups that remain, which is
    how the ablations are computed from a single full run.
    """
    groups = group_by_result(pool, allowed)
    vote = groups[0].representative if groups else None
    if not groups:
        return None, None
    present = {g.key for g in groups}
    kept = [
        (m, w)
        for m, w in zip(selection.matches, selection.winners, strict=True)
        if m.first in present and m.second in present
    ]
    if selection.method != "tournament" or not kept:
        return vote, vote
    wins = tally([m for m, _ in kept], [w for _, w in kept])
    best = choose([g for g in groups if g.key in wins] or groups, wins)
    return (best.representative if best else vote), vote


_BOXED_RE = re.compile(r"\\boxed\{\s*(?:candidate\s*)?(\d+)\s*\}", re.IGNORECASE)
_LOOSE_RE = re.compile(r"(?:candidate|answer|select(?:ed)?|choose|choice)\W{0,3}(?:is\W{0,3})?#?(\d+)", re.I)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def parse_choice(text: str, n_options: int = 2, *, strict: bool = True) -> int | None:
    """The candidate label the selector chose, 1-based, or ``None``.

    The answer is the last ``\\boxed{n}``. With ``strict=False`` a final
    "candidate n" / "answer: n" also counts, which helps untrained models.
    """
    body = _THINK_RE.sub("", text or "")
    found = _BOXED_RE.findall(body)
    if not found and not strict:
        found = _LOOSE_RE.findall(body[-300:])
    if not found:
        return None
    choice = int(found[-1])
    return choice if 1 <= choice <= n_options else None
