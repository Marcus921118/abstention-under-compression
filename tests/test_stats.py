"""Bootstrap CI sanity tests."""

import numpy as np
import pytest

from aq.analysis.stats import bootstrap_ci


def test_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(0)
    data = rng.normal(0.5, 0.1, 500)
    point, lo, hi = bootstrap_ci(data, n_boot=2000, seed=1)
    assert lo < point < hi
    assert abs(point - data.mean()) < 1e-9
    assert (hi - lo) < 0.05  # tight CI for n=500


def test_bootstrap_ci_empty():
    point, lo, hi = bootstrap_ci([])
    assert np.isnan(point) and np.isnan(lo) and np.isnan(hi)


def _rec(i, answerable, abstained, correct=None, tok=0.7):
    return {
        "sample_id": f"s{i}", "answerable": answerable, "abstained": abstained,
        "correct": correct, "verbalized_confidence": None,
        "meta": {"token_prob_confidence": tok},
    }


def test_paired_bootstrap_detects_abstention_drop():
    from aq.analysis.stats import paired_bootstrap_diff

    # 20 unanswerable + 20 answerable, same sample_ids in hi and lo.
    hi, lo = [], []
    for i in range(20):  # unanswerable
        hi.append(_rec(i, False, True))            # hi abstains (good)
        lo.append(_rec(i, False, i < 4))           # lo rarely abstains (bad)
    for i in range(20, 40):  # answerable
        hi.append(_rec(i, True, False, correct=True))
        lo.append(_rec(i, True, False, correct=True))
    r = paired_bootstrap_diff(hi, lo, "abstention_f1", n_boot=500, seed=0)
    assert r["n"] == 40
    assert r["diff_lo_minus_hi"] < 0          # abstention F1 dropped
    assert r["ci"][1] < 0                      # CI entirely below 0
    assert r["p_two_sided"] < 0.05             # significant


def test_paired_bootstrap_p_never_exceeds_one():
    """Known-answer case for the 2026-07-26 p>1 bug.

    When hi and lo are IDENTICAL the bootstrap difference is exactly 0 in every
    resample. Both tails (>=0 and <=0) are then 1.0, so the unclipped formula
    2 * min(1.0, 1.0) = 2.0. The correct answer is p = 1.0 (no evidence of any
    difference whatsoever) -- a p-value is a probability and cannot exceed 1.

    Regression guard: results/significance_report.md once contained p=1.116.
    """
    from aq.analysis.stats import paired_bootstrap_diff

    hi, lo = [], []
    for i in range(20):
        hi.append(_rec(i, False, True))
        lo.append(_rec(i, False, True))          # identical to hi
    for i in range(20, 40):
        hi.append(_rec(i, True, False, correct=True))
        lo.append(_rec(i, True, False, correct=True))

    for metric in ("abstention_f1", "accuracy_on_answered",
                   "over_confident_wrong_rate"):
        r = paired_bootstrap_diff(hi, lo, metric, n_boot=200, seed=0)
        p = r["p_two_sided"]
        if np.isnan(p):
            continue
        assert 0.0 <= p <= 1.0, f"{metric}: p={p} outside [0,1]"
        assert r["diff_lo_minus_hi"] == pytest.approx(0.0, abs=1e-12)
        assert p == pytest.approx(1.0), f"{metric}: identical arms should give p=1.0"


def test_t_two_sided_p_against_standard_tables():
    """Known-answer case for the scipy-free Student-t tail.

    Critical values from standard two-sided t tables. If this drifts, every
    seed-level p-value in the project (FDR re-run, autopsy) drifts with it.
    """
    from aq.analysis.stats import t_two_sided_p

    for t, df, expected in [(2.776, 4, 0.05), (2.132, 4, 0.10),
                            (4.604, 4, 0.01), (2.571, 5, 0.05),
                            (12.706, 1, 0.05), (1.960, 10000, 0.05)]:
        got = t_two_sided_p(t, df)
        assert abs(got - expected) < 0.002, f"t={t}, df={df}: {got} != {expected}"

    assert t_two_sided_p(0.0, 4) == pytest.approx(1.0)
    assert 0.0 <= t_two_sided_p(1e6, 4) <= 1.0
    assert np.isnan(t_two_sided_p(float("nan"), 4))
    assert np.isnan(t_two_sided_p(1.0, 0))


def test_t_critical_and_mean_ci():
    from aq.analysis.stats import mean_ci, t_critical

    # standard two-sided .05 critical values
    for df, expected in [(1, 12.706), (4, 2.776), (5, 2.571), (10, 2.228),
                         (30, 2.042)]:
        assert t_critical(df) == pytest.approx(expected, abs=0.002), f"df={df}"

    # CI must bracket the mean and widen as the spread grows
    m, lo, hi = mean_ci([0.10, 0.11, 0.09, 0.10, 0.10])
    assert lo < m < hi and m == pytest.approx(0.10, abs=1e-9)
    _, lo2, hi2 = mean_ci([0.10, 0.30, -0.10, 0.20, 0.00])
    assert (hi2 - lo2) > (hi - lo)

    # identical values -> zero-width interval, not a crash
    m3, lo3, hi3 = mean_ci([0.2, 0.2, 0.2])
    assert lo3 == pytest.approx(0.2) and hi3 == pytest.approx(0.2)

    # too few points -> nan bounds rather than a bogus interval
    _, lo4, hi4 = mean_ci([1.0])
    assert np.isnan(lo4) and np.isnan(hi4)


def test_one_sample_t_zero_variance_and_short_input():
    from aq.analysis.stats import one_sample_t

    t, p, df = one_sample_t([0.2, 0.2, 0.2, 0.2])   # identical, non-zero mean
    assert np.isinf(t) and t > 0 and p == 0.0 and df == 3

    t, p, df = one_sample_t([0.0, 0.0, 0.0])        # identical, zero mean
    assert t == 0.0 and p == 1.0 and df == 2

    t, p, df = one_sample_t([1.0])                  # too few
    assert np.isnan(t) and np.isnan(p)

    # reproduces the value recorded in results/squad2_headline_reinforcement.json
    t, p, df = one_sample_t([0.1997, 0.1908, 0.1973, 0.1958, 0.2161])
    assert t == pytest.approx(46.557, abs=0.01)
    assert df == 4


def test_benjamini_hochberg():
    from aq.analysis.stats import benjamini_hochberg

    # one tiny p among large ones: BH should still reject the tiny one
    q, reject = benjamini_hochberg([0.001, 0.5, 0.6, 0.7, 0.8])
    assert reject[0] and not reject[1:].any()
    assert q[0] == pytest.approx(0.005)  # 0.001 * 5 / 1
    # a borderline p=0.049 among many nulls should NOT survive
    q2, rej2 = benjamini_hochberg([0.049] + [0.5] * 39)
    assert not rej2.any()
    # NaN carried through, never rejected
    q3, rej3 = benjamini_hochberg([float("nan"), 0.01])
    assert np.isnan(q3[0]) and not rej3[0]


def test_paired_bootstrap_no_change_not_significant():
    from aq.analysis.stats import paired_bootstrap_diff

    hi = [_rec(i, False, True) for i in range(10)]
    hi += [_rec(i, True, False, True) for i in range(10, 20)]
    lo = [dict(r) for r in hi]  # identical
    r = paired_bootstrap_diff(hi, lo, "abstention_f1", n_boot=300, seed=1)
    assert r["diff_lo_minus_hi"] == 0.0
    assert r["p_two_sided"] > 0.05
