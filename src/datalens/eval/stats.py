"""Statistics for comparing models on the same questions.

Every model answers the same benchmark questions, so comparisons are
*paired*: McNemar's test looks only at questions where exactly one model is
right, and the paired bootstrap resamples questions, not models. With several
small-vs-large pairs tested at once, p-values get a Holm-Bonferroni correction.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np

from datalens.inference.voting import majority_index


def bootstrap_ci(
    values: Sequence[float], *, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean of ``values``."""
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, data.size, size=(n_boot, data.size))
    means = data[idx].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


@dataclass(frozen=True)
class PairedComparison:
    """Model A vs model B on the same questions (A minus B)."""

    n: int
    acc_a: float
    acc_b: float
    diff: float
    diff_ci: tuple[float, float]
    only_a: int  # questions only A answers correctly
    only_b: int  # questions only B answers correctly
    p_value: float


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar test (binomial test on the discordant pairs)."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_comparison(
    correct_a: Sequence[bool],
    correct_b: Sequence[bool],
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> PairedComparison:
    a = np.asarray(correct_a, dtype=float)
    b = np.asarray(correct_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired comparison needs results for the same questions")
    only_a = int(((a == 1) & (b == 0)).sum())
    only_b = int(((a == 0) & (b == 1)).sum())
    return PairedComparison(
        n=int(a.size),
        acc_a=float(a.mean()) if a.size else math.nan,
        acc_b=float(b.mean()) if b.size else math.nan,
        diff=float((a - b).mean()) if a.size else math.nan,
        diff_ci=bootstrap_ci(a - b, n_boot=n_boot, alpha=alpha, seed=seed),
        only_a=only_a,
        only_b=only_b,
        p_value=mcnemar_exact(only_a, only_b),
    )


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p_values[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from ``n`` samples with ``c`` correct (Chen et al., 2021)."""
    if k > n:
        raise ValueError(f"k={k} exceeds the {n} samples available")
    if n - c < k:
        return 1.0
    return 1.0 - math.prod(1.0 - k / i for i in range(n - c + 1, n + 1))


def maj_at_k(
    keys: Sequence[Hashable | None],
    correct: Sequence[bool],
    empty: Sequence[bool],
    k: int,
    *,
    num_draws: int = 32,
    seed: int = 0,
) -> float:
    """Expected self-consistency accuracy with ``k`` of the available samples.

    Draws ``num_draws`` random size-``k`` subsets and averages whether the
    majority-voted answer is correct, which is less noisy than using only the
    first ``k`` samples. With ``k`` equal to all samples there is one subset.
    """
    n = len(keys)
    if k > n:
        raise ValueError(f"k={k} exceeds the {n} samples available")
    rng = np.random.default_rng(seed)
    draws = 1 if k == n else num_draws
    hits = 0
    for _ in range(draws):
        subset = sorted(rng.choice(n, size=k, replace=False)) if k < n else list(range(n))
        choice = majority_index([keys[i] for i in subset], [empty[i] for i in subset])
        hits += int(choice is not None and correct[subset[choice]])
    return hits / draws
