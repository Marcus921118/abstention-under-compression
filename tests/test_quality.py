"""Tests for output-quality confound controls."""

from aq.eval.quality import follows_confidence_format, is_degenerate, n_words, quality_flags


def test_follows_confidence_format():
    assert follows_confidence_format("Paris.\nConfidence: 90%")
    assert follows_confidence_format("confidence = 12.5 %")
    assert not follows_confidence_format("I think it is Paris.")
    assert not follows_confidence_format("")


def test_is_degenerate():
    assert is_degenerate("")
    assert is_degenerate("a")
    assert is_degenerate("yes " * 20)               # heavy repetition
    assert is_degenerate("loooo" + "o" * 40)        # long char run
    assert not is_degenerate("The capital of France is Paris.")
    assert not is_degenerate("I don't know the answer to this question.")


def test_n_words_and_flags():
    assert n_words("one two three") == 3
    f = quality_flags("Answer: Paris.\nConfidence: 80%")
    assert f["format_ok"] is True and f["degenerate"] is False and f["n_words"] >= 3
