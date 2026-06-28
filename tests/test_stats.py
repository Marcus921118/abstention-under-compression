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
