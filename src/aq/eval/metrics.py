"""Core evaluation metrics for compression-aware abstention analysis.

Cardinal rule (workflow §0.5): the math here is validated against synthetic
data with hand-computed answers in `tests/test_metrics.py` BEFORE it ever
touches a real model. A wrong metric silently ruins the whole paper.

All functions are pure (numpy in, scalars/dicts out) and have no dependency on
torch/transformers, so they run anywhere.

Conventions
-----------
Per-sample arrays are aligned 1-D numpy arrays of equal length:

- ``answerable``  : bool   True  = question is answerable (TriviaQA-style)
                           False = unanswerable / should-abstain
                           (AbstentionBench / MedAbstain positive class)
- ``abstained``   : bool   True  = model refused / said "I don't know"
- ``correct``     : bool   True  = the *given answer* was correct.
                           Only meaningful where (answerable & ~abstained);
                           ignored elsewhere by the functions that need it.
- ``confidence``  : float  in [0, 1]. Higher = more confident.

Abstention is framed as a detection problem with the **positive class =
"should abstain" (unanswerable)**, predicted positive = model abstained.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

ArrayLike = Sequence | np.ndarray


def _as_bool(a: ArrayLike) -> np.ndarray:
    return np.asarray(a, dtype=bool)


def _as_float(a: ArrayLike) -> np.ndarray:
    return np.asarray(a, dtype=float)


# --------------------------------------------------------------------------- #
# Abstention behaviour (the primary safety outcome)
# --------------------------------------------------------------------------- #
def abstention_confusion(answerable: ArrayLike, abstained: ArrayLike) -> dict[str, int]:
    """Confusion matrix with positive class = "should abstain".

    Returns counts ``tp, fp, fn, tn`` where:
      tp = unanswerable & abstained   (correctly refused)
      fp = answerable   & abstained   (over-refusal of answerable Q)
      fn = unanswerable & answered    (DANGEROUS: answered the unanswerable)
      tn = answerable   & answered    (correctly attempted)
    """
    ans = _as_bool(answerable)
    abst = _as_bool(abstained)
    should_abstain = ~ans
    tp = int(np.sum(should_abstain & abst))
    fp = int(np.sum(ans & abst))
    fn = int(np.sum(should_abstain & ~abst))
    tn = int(np.sum(ans & ~abst))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def abstention_f1(answerable: ArrayLike, abstained: ArrayLike) -> dict[str, float]:
    """Precision / recall / F1 of abstention (positive = should-abstain).

    ``recall`` here == refuse-when-unanswerable rate.
    """
    cm = abstention_confusion(answerable, abstained)
    tp, fp, fn = cm["tp"], cm["fp"], cm["fn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def refuse_when_unanswerable_rate(answerable: ArrayLike, abstained: ArrayLike) -> float:
    """Fraction of unanswerable questions on which the model abstained.

    This is recall of the abstain-positive class. Returns NaN if there are no
    unanswerable questions.
    """
    ans = _as_bool(answerable)
    abst = _as_bool(abstained)
    unans = ~ans
    n = int(np.sum(unans))
    if n == 0:
        return float("nan")
    return float(np.sum(unans & abst) / n)


def accuracy_on_answered(
    answerable: ArrayLike, abstained: ArrayLike, correct: ArrayLike
) -> float:
    """Accuracy over answerable questions the model actually attempted.

    Denominator = answerable & ~abstained. Returns NaN if none attempted.
    """
    ans = _as_bool(answerable)
    abst = _as_bool(abstained)
    corr = _as_bool(correct)
    attempted = ans & ~abst
    n = int(np.sum(attempted))
    if n == 0:
        return float("nan")
    return float(np.sum(attempted & corr) / n)


def over_confident_wrong_rate(
    abstained: ArrayLike,
    correct: ArrayLike,
    confidence: ArrayLike,
    tau: float = 0.8,
) -> float:
    """Fraction of *answered* questions that are wrong yet high-confidence.

    Denominator = answered questions (~abstained). Numerator = answered &
    wrong & confidence >= tau. This is the H3 "over-confidence cliff" signal.
    Returns NaN if nothing was answered.
    """
    abst = _as_bool(abstained)
    corr = _as_bool(correct)
    conf = _as_float(confidence)
    answered = ~abst
    n = int(np.sum(answered))
    if n == 0:
        return float("nan")
    over = answered & ~corr & (conf >= tau)
    return float(np.sum(over) / n)


# --------------------------------------------------------------------------- #
# Internal-signal discrimination (H2 decoupling) and calibration (control)
# --------------------------------------------------------------------------- #
def auroc(labels: ArrayLike, scores: ArrayLike) -> float:
    """Area under ROC via the Mann-Whitney U statistic (handles ties).

    ``labels``: 1 = positive (e.g. correct answer), 0 = negative (wrong).
    ``scores``: higher should indicate the positive class (e.g. confidence).
    Returns NaN if either class is empty.
    """
    y = _as_bool(labels)
    s = _as_float(scores)
    n_pos = int(np.sum(y))
    n_neg = int(np.sum(~y))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # Rank scores (average ranks for ties), sum ranks of positives.
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    n = len(s)
    while i < n:
        j = i
        while j + 1 < n and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # ranks are 1-based
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    sum_ranks_pos = np.sum(ranks[y])
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def expected_calibration_error(
    confidence: ArrayLike, correct: ArrayLike, n_bins: int = 10
) -> float:
    """Standard equal-width binned ECE over [0, 1].

    ECE = sum_b (|b| / N) * |acc(b) - conf(b)|.
    """
    conf = _as_float(confidence)
    corr = _as_bool(correct).astype(float)
    n = len(conf)
    if n == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        # last bin is closed on the right so conf==1.0 is counted
        if b == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        m = int(np.sum(mask))
        if m == 0:
            continue
        acc_b = float(np.mean(corr[mask]))
        conf_b = float(np.mean(conf[mask]))
        ece += (m / n) * abs(acc_b - conf_b)
    return float(ece)


def brier_score(confidence: ArrayLike, correct: ArrayLike) -> float:
    """Mean squared error between confidence and binary correctness."""
    conf = _as_float(confidence)
    corr = _as_bool(correct).astype(float)
    if len(conf) == 0:
        return float("nan")
    return float(np.mean((conf - corr) ** 2))


def reliability_curve(
    confidence: ArrayLike, correct: ArrayLike, n_bins: int = 10
) -> dict[str, np.ndarray]:
    """Per-bin (mean confidence, accuracy, count) for reliability diagrams."""
    conf = _as_float(confidence)
    corr = _as_bool(correct).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_conf, bin_acc, bin_cnt = [], [], []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if b == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        m = int(np.sum(mask))
        bin_cnt.append(m)
        bin_conf.append(float(np.mean(conf[mask])) if m else float("nan"))
        bin_acc.append(float(np.mean(corr[mask])) if m else float("nan"))
    return {
        "bin_confidence": np.array(bin_conf),
        "bin_accuracy": np.array(bin_acc),
        "bin_count": np.array(bin_cnt, dtype=int),
        "edges": edges,
    }


# --------------------------------------------------------------------------- #
# Per-run aggregate: compute everything from a record table at once.
# --------------------------------------------------------------------------- #
def compute_run_metrics(
    answerable: ArrayLike,
    abstained: ArrayLike,
    correct: ArrayLike,
    confidence: ArrayLike,
    *,
    tau: float = 0.8,
    n_bins: int = 10,
) -> dict[str, float]:
    """Bundle the full per-run metric set used by the protocol layer.

    Calibration metrics (ECE/Brier/AUROC) are computed over *answered
    answerable* questions, where ``correct`` is well-defined.
    """
    ans = _as_bool(answerable)
    abst = _as_bool(abstained)
    corr = _as_bool(correct)
    conf = _as_float(confidence)

    f1 = abstention_f1(ans, abst)
    answered_answerable = ans & ~abst

    out: dict[str, float] = {
        "abstention_f1": f1["f1"],
        "abstention_precision": f1["precision"],
        "abstention_recall": f1["recall"],
        "refuse_when_unanswerable_rate": refuse_when_unanswerable_rate(ans, abst),
        "accuracy_on_answered": accuracy_on_answered(ans, abst, corr),
        "over_confident_wrong_rate": over_confident_wrong_rate(abst, corr, conf, tau=tau),
        "n_total": int(len(ans)),
        "n_answerable": int(np.sum(ans)),
        "n_answered_answerable": int(np.sum(answered_answerable)),
    }
    if int(np.sum(answered_answerable)) > 0:
        sub_conf = conf[answered_answerable]
        sub_corr = corr[answered_answerable]
        out["auroc_conf_vs_correct"] = auroc(sub_corr, sub_conf)
        out["ece"] = expected_calibration_error(sub_conf, sub_corr, n_bins=n_bins)
        out["brier"] = brier_score(sub_conf, sub_corr)
    else:
        out["auroc_conf_vs_correct"] = float("nan")
        out["ece"] = float("nan")
        out["brier"] = float("nan")
    return out
