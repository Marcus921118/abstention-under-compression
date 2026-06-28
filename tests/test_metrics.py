"""Known-answer unit tests for the metric math (workflow §0.5, §6).

These are the most important tests in the project: every value below is
hand-computed so a regression in the metric layer is caught before it can
corrupt a real experiment.
"""

import math

import numpy as np
import pytest

from aq.eval import metrics as M
from aq.eval import protocol as P


# --------------------------------------------------------------------------- #
# Abstention confusion / F1
# --------------------------------------------------------------------------- #
def test_abstention_confusion_counts():
    # answerable:  A  A  U  U   (U = unanswerable, positive class)
    # abstained:   F  T  T  F
    answerable = [True, True, False, False]
    abstained = [False, True, True, False]
    cm = M.abstention_confusion(answerable, abstained)
    # tp: U&abst -> idx2                 = 1
    # fp: A&abst -> idx1                 = 1
    # fn: U&~abst -> idx3                = 1
    # tn: A&~abst -> idx0                = 1
    assert cm == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}


def test_abstention_f1_known():
    # 3 unanswerable (2 refused, 1 answered), 2 answerable (1 wrongly refused)
    answerable = [False, False, False, True, True]
    abstained = [True, True, False, True, False]
    r = M.abstention_f1(answerable, abstained)
    # tp=2, fp=1, fn=1
    # precision = 2/3, recall = 2/3, f1 = 2/3
    assert r["precision"] == pytest.approx(2 / 3)
    assert r["recall"] == pytest.approx(2 / 3)
    assert r["f1"] == pytest.approx(2 / 3)


def test_refuse_when_unanswerable_rate():
    answerable = [False, False, False, False, True]
    abstained = [True, True, True, False, False]
    # 3/4 unanswerable refused
    assert M.refuse_when_unanswerable_rate(answerable, abstained) == pytest.approx(0.75)


def test_refuse_rate_nan_when_no_unanswerable():
    assert math.isnan(M.refuse_when_unanswerable_rate([True, True], [False, True]))


# --------------------------------------------------------------------------- #
# Accuracy / over-confidence
# --------------------------------------------------------------------------- #
def test_accuracy_on_answered():
    answerable = [True, True, True, True, False]
    abstained = [False, False, False, True, False]  # one answerable abstained -> excluded
    correct = [True, True, False, False, False]
    # attempted answerable = idx0,1,2 -> correct 2/3
    assert M.accuracy_on_answered(answerable, abstained, correct) == pytest.approx(2 / 3)


def test_over_confident_wrong_rate():
    # answered all; wrong&high-conf cases define numerator
    abstained = [False, False, False, False]
    correct = [False, False, True, True]
    confidence = [0.9, 0.7, 0.95, 0.4]
    # answered = 4; over-conf wrong (>=0.8 & wrong): idx0 only -> 1/4
    r = M.over_confident_wrong_rate(abstained, correct, confidence, tau=0.8)
    assert r == pytest.approx(0.25)


def test_over_confident_wrong_rate_excludes_abstained():
    abstained = [True, False]
    correct = [False, False]
    confidence = [0.99, 0.99]
    # only idx1 answered; it's wrong & high-conf -> 1/1
    r = M.over_confident_wrong_rate(abstained, correct, confidence, tau=0.8)
    assert r == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# AUROC (Mann-Whitney) -- cross-checked against sklearn value
# --------------------------------------------------------------------------- #
def test_auroc_known_value():
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.4, 0.35, 0.8]
    # pairwise: pos{0.35,0.8} vs neg{0.1,0.4}: 0.35>0.1,0.8>0.1,0.8>0.4 (3), 0.35<0.4
    # => 3/4 = 0.75
    assert M.auroc(labels, scores) == pytest.approx(0.75)


def test_auroc_perfect_and_inverted():
    assert M.auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert M.auroc([0, 0, 1, 1], [0.8, 0.9, 0.1, 0.2]) == pytest.approx(0.0)


def test_auroc_ties_give_half():
    # all scores equal -> every pair is a tie -> 0.5
    assert M.auroc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)


def test_auroc_nan_single_class():
    assert math.isnan(M.auroc([1, 1, 1], [0.1, 0.2, 0.3]))


def test_auroc_matches_sklearn_random():
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = rng.integers(0, 2, size=50)
        if y.min() == y.max():
            continue
        s = rng.random(50)
        assert M.auroc(y, s) == pytest.approx(sklearn_metrics.roc_auc_score(y, s))


# --------------------------------------------------------------------------- #
# ECE / Brier
# --------------------------------------------------------------------------- #
def test_ece_perfectly_calibrated_is_zero():
    # bin of conf=0.0 all wrong, bin of conf=1.0 all correct -> ECE 0
    confidence = [0.0, 0.0, 1.0, 1.0]
    correct = [False, False, True, True]
    assert M.expected_calibration_error(confidence, correct, n_bins=10) == pytest.approx(0.0)


def test_ece_known_value():
    # two items in [0.0,0.1) with conf 0.0, both correct -> |1.0-0.0|=1.0, weight 1.0
    confidence = [0.0, 0.0]
    correct = [True, True]
    assert M.expected_calibration_error(confidence, correct, n_bins=10) == pytest.approx(1.0)


def test_ece_handles_conf_one_in_last_bin():
    confidence = [1.0, 1.0]
    correct = [True, False]  # acc 0.5, conf 1.0 -> |0.5-1.0|=0.5
    assert M.expected_calibration_error(confidence, correct, n_bins=10) == pytest.approx(0.5)


def test_brier_known():
    confidence = [1.0, 0.0, 0.5]
    correct = [True, False, True]
    # (0)^2 + (0)^2 + (0.5)^2 = 0.25 ; /3
    assert M.brier_score(confidence, correct) == pytest.approx(0.25 / 3)


# --------------------------------------------------------------------------- #
# Bundle
# --------------------------------------------------------------------------- #
def test_compute_run_metrics_shape():
    answerable = [True, True, False, False, True]
    abstained = [False, True, True, False, False]
    correct = [True, False, False, False, True]
    confidence = [0.9, 0.2, 0.1, 0.95, 0.7]
    out = M.compute_run_metrics(answerable, abstained, correct, confidence, tau=0.8)
    for k in ("abstention_f1", "refuse_when_unanswerable_rate",
              "accuracy_on_answered", "over_confident_wrong_rate",
              "auroc_conf_vs_correct", "ece", "brier"):
        assert k in out
    assert out["n_total"] == 5
    assert out["n_answerable"] == 3


# --------------------------------------------------------------------------- #
# Protocol indices
# --------------------------------------------------------------------------- #
def test_selective_damage_index():
    base = {"abstention_f1": 0.9, "accuracy_on_answered": 0.80}
    quant = {"abstention_f1": 0.6, "accuracy_on_answered": 0.78}
    r = P.selective_damage_index(base, quant)
    # f1 drop 0.3, acc drop 0.02 -> 15.0
    assert r["abstention_f1_drop"] == pytest.approx(0.3)
    assert r["accuracy_drop"] == pytest.approx(0.02)
    assert r["selective_damage_index"] == pytest.approx(15.0)


def test_selective_damage_index_nan_when_acc_unchanged():
    base = {"abstention_f1": 0.9, "accuracy_on_answered": 0.80}
    quant = {"abstention_f1": 0.6, "accuracy_on_answered": 0.80}
    r = P.selective_damage_index(base, quant)
    assert math.isnan(r["selective_damage_index"])
    assert r["abstention_f1_drop"] == pytest.approx(0.3)


def test_decoupling_index():
    base = {"abstention_f1": 1.0, "auroc_conf_vs_correct": 1.0}
    quant = {"abstention_f1": 0.5, "auroc_conf_vs_correct": 0.9}
    r = P.decoupling_index(base, quant)
    # behaviour rel drop 0.5, signal rel drop 0.1 -> decoupling 0.4
    assert r["behaviour_rel_drop"] == pytest.approx(0.5)
    assert r["signal_rel_drop"] == pytest.approx(0.1)
    assert r["decoupling_index"] == pytest.approx(0.4)


def test_overconfidence_cliff():
    bits = [16, 8, 4, 3, 2]
    over = [0.02, 0.03, 0.05, 0.30, 0.45]  # big jump 4->3
    res = P.find_overconfidence_cliff(bits, over)
    assert res.cliff_bits == 3
    assert res.max_jump == pytest.approx(0.25)


def test_overconfidence_cliff_no_rise():
    bits = [16, 8, 4]
    over = [0.5, 0.3, 0.1]  # monotonically decreasing -> no cliff
    res = P.find_overconfidence_cliff(bits, over)
    assert res.cliff_bits is None
