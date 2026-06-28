"""Statistical significance: paired tests + bootstrap CIs (workflow §6).

bootstrap_ci is implemented now (pure numpy, testable). Paired significance
tests over matched per-question records come in Stage 3.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


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
        p = 2.0 * min((finite >= 0).mean(), (finite <= 0).mean())
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
