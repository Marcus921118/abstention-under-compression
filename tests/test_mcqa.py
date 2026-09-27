"""Tests for judge-free MCQA scoring."""

from aq.eval.mcqa import extract_choice, mcqa_outcome


def test_extract_choice():
    assert extract_choice("The answer is B.") == "B"
    assert extract_choice("(C)") == "C"
    assert extract_choice("D") == "D"
    assert extract_choice("I think A) is correct") == "A"
    assert extract_choice("no letter here at all") is None
    assert extract_choice("") is None


def test_mcqa_outcome_answerable():
    # gold is a content letter; model picks it -> correct, not abstained
    r = mcqa_outcome("Answer: B", gold_letter="B", abstain_letter="D")
    assert r["abstained"] is False and r["correct"] is True
    # picks wrong content letter -> incorrect
    r = mcqa_outcome("A", gold_letter="B", abstain_letter="D")
    assert r["abstained"] is False and r["correct"] is False
    # picks the abstain option on an answerable item -> over-abstention
    r = mcqa_outcome("D", gold_letter="B", abstain_letter="D")
    assert r["abstained"] is True and r["correct"] is None


def test_mcqa_outcome_unanswerable():
    # gold == abstain letter; picking it = correctly abstained
    r = mcqa_outcome("The answer is D", gold_letter="D", abstain_letter="D")
    assert r["abstained"] is True
    # answering a content letter on an unanswerable item = failure to abstain
    r = mcqa_outcome("B", gold_letter="D", abstain_letter="D")
    assert r["abstained"] is False


def test_mcqa_outcome_ghost_abstention():
    # No letter chosen + abstain phrase -> should be abstained, not "answered wrong"
    r = mcqa_outcome("I don't know.\nConfidence: 0%", gold_letter="B", abstain_letter="D")
    assert r["abstained"] is True and r["correct"] is None and r["choice"] is None
    r = mcqa_outcome("I cannot determine the answer.", gold_letter="A", abstain_letter="D")
    assert r["abstained"] is True and r["correct"] is None
    # No letter + no abstain phrase (degenerate output) -> still answered, not abstained
    r = mcqa_outcome("random gibberish output", gold_letter="B", abstain_letter="D")
    assert r["abstained"] is False and r["correct"] is False
    # Has letter + abstain phrase -> letter wins (model hedged but still picked)
    r = mcqa_outcome("I'm not sure, but B", gold_letter="B", abstain_letter="D")
    assert r["abstained"] is False and r["correct"] is True
