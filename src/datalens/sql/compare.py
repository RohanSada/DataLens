"""Result-set comparison: BIRD execution accuracy and Soft-F1.

Both follow the official BIRD evaluation scripts so numbers are comparable with
the leaderboard:

* EX treats results as *sets* of rows: order and duplicates are ignored.
* Soft-F1 (introduced with BIRD mini-dev) gives partial credit per cell.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

Row = tuple[Any, ...]


def _row_key(row: Row) -> str:
    return repr(row)


def _normalise_value(value: Any) -> Any:
    # Python treats 1 == 1.0 as equal in sets (so does EX); make the hash agree.
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def result_key(rows: Sequence[Row]) -> str:
    """Stable digest of a result as a set of rows, for grouping and voting."""
    canonical = sorted({repr(tuple(_normalise_value(v) for v in row)) for row in rows})
    return hashlib.sha1("\n".join(canonical).encode("utf-8")).hexdigest()[:16]


def exec_match(pred_rows: Sequence[Row], gold_rows: Sequence[Row]) -> bool:
    """Official BIRD execution-accuracy criterion: ``set(pred) == set(gold)``."""
    return set(pred_rows) == set(gold_rows)


def _row_match(pred_row: Row, gold_row: Row) -> tuple[float, float, float]:
    total = len(gold_row)
    if total == 0:
        return 0.0, 0.0, 0.0
    matches = sum(1 for value in pred_row if value in gold_row)
    pred_only = sum(1 for value in pred_row if value not in gold_row)
    gold_only = sum(1 for value in gold_row if value not in pred_row)
    return matches / total, pred_only / total, gold_only / total


def soft_f1(pred_rows: Sequence[Row], gold_rows: Sequence[Row]) -> float:
    """Soft-F1 from the BIRD mini-dev evaluation script.

    Duplicate rows are dropped, rows are paired by position, and each pair scores
    the share of matching cells. Unpaired rows count as fully wrong. The official
    script pairs rows in Python ``set`` iteration order, which is not stable across
    runs; here rows are sorted by ``repr`` first so the score is reproducible.
    """
    if not pred_rows and not gold_rows:
        return 1.0
    pred = sorted(set(pred_rows), key=_row_key)
    gold = sorted(set(gold_rows), key=_row_key)

    tp = fp = fn = 0.0
    for i, gold_row in enumerate(gold):
        if i >= len(pred):
            fn += 1.0
            continue
        match, pred_only, gold_only = _row_match(pred[i], gold_row)
        tp += match
        fp += pred_only
        fn += gold_only
    fp += max(0, len(pred) - len(gold))

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def ves_reward(time_ratio: float) -> float:
    """Reward-based VES (R-VES) bucket for one correct query.

    ``time_ratio`` is gold execution time divided by predicted execution time, so
    values above 1 mean the prediction ran faster than the reference query.
    """
    if time_ratio <= 0:
        return 0.0
    if time_ratio >= 2:
        return 1.25
    if time_ratio >= 1:
        return 1.0
    if time_ratio >= 0.5:
        return 0.75
    if time_ratio >= 0.25:
        return 0.5
    return 0.25
