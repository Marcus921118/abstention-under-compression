"""Tests for the answer-correctness scorer (pure, no model)."""

from aq.eval.correctness import is_correct, normalize


def test_normalize_strips_articles_punct_case():
    assert normalize("The  Paris, France!") == "paris france"
    assert normalize("H2O") == "h2o"


def test_is_correct_substring_match():
    assert is_correct("The capital is Paris.", ["Paris"])
    assert is_correct("I think the answer is 4.", ["4", "four"])
    assert is_correct("It ended in 1945 after the war.", ["1945"])


def test_is_correct_word_boundary():
    # "art" should not match inside "Mars"
    assert not is_correct("The planet is Mars.", ["art"])
    assert is_correct("The planet is Mars.", ["Mars"])


def test_is_correct_no_gold_is_false():
    assert not is_correct("anything", [])


def test_is_correct_case_accent_insensitive():
    # accent stripping is intentional: "MÄRS" -> "mars" matches "mars"
    assert is_correct("the answer is MÄRS", ["mars"])
    assert is_correct("Montréal is the city", ["montreal"])
    # genuinely different word still does not match
    assert not is_correct("the answer is Venus", ["mars"])
