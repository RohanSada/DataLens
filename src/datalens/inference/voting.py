"""Execution-based self-consistency.

Sample several queries, run them, and group them by the result they return. The
answer is the largest group: queries that disagree in text but agree in result
are counted as votes for the same answer. Failed queries get no vote.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class VoteGroup:
    """Candidates that produced the same result."""

    key: Hashable
    members: tuple[int, ...]
    empty_result: bool

    @property
    def votes(self) -> int:
        return len(self.members)

    @property
    def representative(self) -> int:
        return self.members[0]


def group_candidates(keys: Sequence[Hashable | None], empty: Sequence[bool] | None = None) -> list[VoteGroup]:
    """Group candidate indices by result key, best group first.

    ``keys[i]`` is the result fingerprint of candidate ``i`` or ``None`` if it
    failed. Groups are ordered by votes, then non-empty results before empty ones
    (an empty answer is the most common way to be confidently wrong), then by the
    earliest member so the order is deterministic.
    """
    empty = empty if empty is not None else [False] * len(keys)
    members: dict[Hashable, list[int]] = {}
    for i, key in enumerate(keys):
        if key is not None:
            members.setdefault(key, []).append(i)
    groups = [
        VoteGroup(key=key, members=tuple(idx), empty_result=bool(empty[idx[0]]))
        for key, idx in members.items()
    ]
    groups.sort(key=lambda g: (-g.votes, g.empty_result, g.representative))
    return groups


def majority_index(keys: Sequence[Hashable | None], empty: Sequence[bool] | None = None) -> int | None:
    """Index of the candidate self-consistency selects, or ``None`` if all failed."""
    groups = group_candidates(keys, empty)
    return groups[0].representative if groups else None
