from __future__ import annotations

import math

import pytest

from datalens.eval.stats import (
    bootstrap_ci,
    holm_adjust,
    maj_at_k,
    mcnemar_exact,
    paired_comparison,
    pass_at_k,
)


def test_bootstrap_ci_brackets_the_mean():
    values = [1.0] * 70 + [0.0] * 30
    lo, hi = bootstrap_ci(values, n_boot=2000)
    assert lo < 0.7 < hi
    assert lo > 0.55 and hi < 0.85
    assert all(math.isnan(x) for x in bootstrap_ci([]))


def test_mcnemar_exact_known_values():
    assert mcnemar_exact(0, 0) == 1.0
    # 10 discordant pairs all favouring A: p = 2 * 0.5**10
    assert mcnemar_exact(10, 0) == pytest.approx(2 * 0.5**10)
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(3, 7) == pytest.approx(0.34375)


def test_paired_comparison_counts_discordant_pairs():
    a = [True, True, False, True, False]
    b = [True, False, False, False, True]
    comp = paired_comparison(a, b, n_boot=500)
    assert (comp.only_a, comp.only_b) == (2, 1)
    assert comp.diff == pytest.approx(0.2)
    assert comp.acc_a == pytest.approx(0.6)
    with pytest.raises(ValueError):
        paired_comparison([True], [True, False])


def test_holm_adjust():
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_holm_is_monotone_and_capped():
    adjusted = holm_adjust([0.001, 0.2, 0.6])
    assert adjusted == pytest.approx([0.003, 0.4, 0.6])


def test_pass_at_k():
    assert pass_at_k(10, 0, 1) == 0.0
    assert pass_at_k(10, 10, 5) == 1.0
    assert pass_at_k(10, 3, 1) == pytest.approx(0.3)
    assert pass_at_k(4, 1, 2) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        pass_at_k(2, 1, 3)


def test_maj_at_k():
    keys = ["right", "wrong", "right", "right"]
    correct = [True, False, True, True]
    empty = [False] * 4
    assert maj_at_k(keys, correct, empty, 4) == 1.0
    assert maj_at_k(keys, correct, empty, 1) == pytest.approx(0.75, abs=0.2)
    assert maj_at_k([None, None], [False, False], [False, False], 2) == 0.0
