"""Statistical significance: paired tests + bootstrap CIs (workflow §6).

bootstrap_ci is implemented now (pure numpy, testable). Paired significance
tests over matched per-question records come in Stage 3.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# Student-t tail probability, no scipy.
#
# Lives here (not in an experiment script) because more than one analysis needs
# it: experiments/fdr_rerun.py and experiments/autopsy_confidence_inflation.py.
# Two copies of the same special function is how one of them silently drifts.
# Known-answer tests against standard tables: tests/test_stats.py.
# --------------------------------------------------------------------------- #

def _betacf(a: float, b: float, x: float, itmax: int = 200,
            eps: float = 3e-16) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log(1.0 - x))
    bt = math.exp(lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t: float, df: int) -> float:
    """P(|T_df| >= |t|).  P(T>t) = 0.5 * I_{df/(df+t^2)}(df/2, 1/2)."""
    if df <= 0 or not math.isfinite(t):
        return float("nan")
    x = df / (df + t * t)
    return min(1.0, _betai(df / 2.0, 0.5, x))


def t_critical(df: int, alpha: float = 0.05) -> float:
    """Two-sided critical value t* with P(|T_df| >= t*) = alpha.

    Inverted from t_two_sided_p by bisection -- no scipy. For df=4, alpha=.05
    this returns 2.776 (see tests/test_stats.py for the table check).
    """
    if df <= 0 or not (0.0 < alpha < 1.0):
        return float("nan")
    lo, hi = 0.0, 1e4
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_two_sided_p(mid, df) > alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def mean_ci(values, alpha: float = 0.05) -> tuple[float, float, float]:
    """(mean, lo, hi) t-based confidence interval for the mean.

    Use when the replicate unit is the seed: the interval must reflect
    seed-level variation, which a within-seed bootstrap cannot see.
    """
    d = np.asarray(values, dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 2:
        return (float(d.mean()) if n else float("nan"),
                float("nan"), float("nan"))
    m = float(d.mean())
    sem = float(d.std(ddof=1) / np.sqrt(n))
    h = t_critical(n - 1, alpha) * sem
    return m, m - h, m + h


def one_sample_t(values, mu: float = 0.0) -> tuple[float, float, int]:
    """One-sample t against mu. Returns (t, two-sided p, df).

    Non-finite entries are dropped. Zero variance is handled explicitly rather
    than producing a divide-by-zero warning and a nan.
    """
    d = np.asarray(values, dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 2:
        return float("nan"), float("nan"), max(n - 1, 0)
    sd = float(d.std(ddof=1))
    centred = float(d.mean() - mu)
    if sd == 0.0:
        return ((float("inf") if centred > 0 else float("-inf")) if centred else 0.0,
                0.0 if centred else 1.0, n - 1)
    t = centred / (sd / math.sqrt(n))
    return t, t_two_sided_p(t, n - 1), n - 1


def benjamini_hochberg(pvalues, alpha: float = 0.05):
    """Benjamini-Hochberg FDR adjustment.

    Returns (qvalues, reject) aligned to the input order. NaN p-values are
    carried through as NaN and never rejected. qvalue[i] is the BH-adjusted
    p-value; reject[i] is True iff qvalue[i] <= alpha.
    """
    p = np.asarray(pvalues, dtype=float)
    finite = np.isfinite(p)
    q = np.full(p.shape, np.nan)
    idx = np.where(finite)[0]
    if idx.size == 0:
        return q, np.zeros(p.shape, dtype=bool)
    pf = p[idx]
    m = pf.size
    order = np.argsort(pf)
    ranked = pf[order]
    adj = np.empty(m)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        prev = min(prev, ranked[i] * m / (i + 1))
        adj[i] = prev
    qf = np.empty(m)
    qf[order] = np.minimum(adj, 1.0)
    q[idx] = qf
    reject = np.zeros(p.shape, dtype=bool)
    reject[idx] = qf <= alpha
    return q, reject


def paired_bootstrap_diff(
    hi_records: list[dict],
    lo_records: list[dict],
    metric: str,
    *,
    signal: str = "token_prob",
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Paired bootstrap of a metric's change from hi-bit to lo-bit on matched items.

    Records are aligned by sample_id (same questions under two precisions). The
    reported diff is metric(lo) - metric(hi): negative = the metric dropped under
    compression (e.g. abstention_f1), positive = it rose (e.g. over-conf-wrong).
    Returns point diff, 95% CI, and a two-sided bootstrap p-value.
    """
    from ..eval import metrics as M  # noqa: PLC0415
    from ..eval.scoring import extract_arrays  # noqa: PLC0415

    metric_fns = {
        "abstention_f1": lambda a, ab, c, cf: M.abstention_f1(a, ab)["f1"],
        "refuse_when_unanswerable_rate":
            lambda a, ab, c, cf: M.refuse_when_unanswerable_rate(a, ab),
        "accuracy_on_answered": lambda a, ab, c, cf: M.accuracy_on_answered(a, ab, c),
        "over_confident_wrong_rate":
            lambda a, ab, c, cf: M.over_confident_wrong_rate(ab, c, cf),
    }
    fn = metric_fns[metric]
    lo_by = {r["sample_id"]: r for r in lo_records}
    pairs = [(h, lo_by[h["sample_id"]]) for h in hi_records if h["sample_id"] in lo_by]
    if not pairs:
        return {"metric": metric, "n": 0, "diff_lo_minus_hi": float("nan"),
                "ci": [float("nan"), float("nan")], "p_two_sided": float("nan")}
    hi = [p[0] for p in pairs]
    lo = [p[1] for p in pairs]
    ha, hab, hc, hcf = (np.asarray(x) for x in extract_arrays(hi, signal))
    la, lab, lc, lcf = (np.asarray(x) for x in extract_arrays(lo, signal))

    def _diff(idx):
        d = fn(la[idx], lab[idx], lc[idx], lcf[idx]) - fn(ha[idx], hab[idx], hc[idx], hcf[idx])
        return d

    n = len(hi)
    all_idx = np.arange(n)
    point = _diff(all_idx)
    rng = np.random.default_rng(seed)
    diffs = np.array([_diff(rng.integers(0, n, n)) for _ in range(n_boot)], dtype=float)
    finite = diffs[np.isfinite(diffs)]
    if finite.size == 0:
        ci = [float("nan"), float("nan")]
        p = float("nan")
    else:
        ci = [float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))]
        # Two-sided bootstrap p. The tails are computed with >= and <=, so a
        # bootstrap distribution centred on 0 counts the point mass at 0 in
        # BOTH tails; doubling then yields p > 1, which is impossible and
        # corrupts the BH step-up when fed to benjamini_hochberg().
        # Observed in the wild: results/significance_report.md had
        # `bnb·mcqa_mmlu·phi3_mini accuracy_on_answered d=+0.000 p=1.116`.
        # Clip to 1.0. (2026-07-26; see docs/agents/CLAIM_AUDIT_20260726.md §4.1)
        p = min(1.0, 2.0 * min((finite >= 0).mean(), (finite <= 0).mean()))
    return {"metric": metric, "n": n, "diff_lo_minus_hi": float(point),
            "ci": ci, "p_two_sided": float(p)}


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI. Returns (point_estimate, lo, hi)."""
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(arr)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    point = float(statistic(arr))
    boots = np.empty(n_boot)
    for i in range(n_boot):
        sample = arr[rng.integers(0, n, size=n)]
        boots[i] = statistic(sample)
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return (point, lo, hi)
