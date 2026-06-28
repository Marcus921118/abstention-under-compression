"""Smoke tests that need no ML stack: parsers, record round-trip, config.

The model-backed smoke test (tiny-gpt2, workflow §6) is gated on transformers
being importable and skips otherwise, so CI stays green on the metrics-only env.
"""

import math

import pytest

from aq.confidence.token_prob import sequence_confidence_from_logprobs
from aq.confidence.verbalized import parse_verbalized_confidence
from aq.data.base import GenerationRecord
from aq.eval.abstain import is_abstention
from aq.eval.pipeline import read_records, write_records


def test_parse_verbalized_confidence():
    assert parse_verbalized_confidence("Answer: Paris.\nConfidence: 90%") == pytest.approx(0.9)
    assert parse_verbalized_confidence("Confidence: 12.5%") == pytest.approx(0.125)
    assert parse_verbalized_confidence("no confidence stated") is None
    # clamps and takes the last value
    assert parse_verbalized_confidence("Confidence: 50%\nConfidence: 150%") == pytest.approx(1.0)


def test_token_prob_confidence():
    # exp(mean([log .5, log .5])) = .5
    lp = [math.log(0.5), math.log(0.5)]
    assert sequence_confidence_from_logprobs(lp) == pytest.approx(0.5)
    assert sequence_confidence_from_logprobs([]) is None


def test_is_abstention():
    assert is_abstention("I don't know the answer.")
    assert is_abstention("There is not enough information to answer.")
    assert is_abstention("Please consult your doctor.")
    assert not is_abstention("The capital of France is Paris.")
    # MCQA explicit abstain option
    assert is_abstention("E", abstain_option_tokens=["E", "cannot be determined"])


def test_is_abstention_hedges():
    # hedges that the G2 slice exposed as misses
    assert is_abstention("I don't have an exact number for the total sand.")
    assert is_abstention("It is impossible to predict tomorrow's closing price.")
    assert is_abstention("I cannot predict the outcome of a future election.")
    assert is_abstention("I would need more information to answer that.")
    assert is_abstention("There is no way to know the current temperature.")
    # genuine confident answers must NOT be flagged
    assert not is_abstention("The Pacific is the largest ocean.")
    assert not is_abstention("World War II ended in 1945.")


def test_record_round_trip(tmp_path):
    recs = [
        GenerationRecord(
            run_id="r1", sample_id="s1", dataset="d", answerable=False,
            category="false_premise", output_text="I don't know", abstained=True,
            correct=None, verbalized_confidence=0.1,
        ),
        GenerationRecord(
            run_id="r1", sample_id="s2", dataset="d", answerable=True,
            category="answerable", output_text="Paris", abstained=False,
            correct=True, verbalized_confidence=0.9,
        ),
    ]
    path = tmp_path / "raw.jsonl"
    assert write_records(path, recs) == 2
    loaded = read_records(path)
    assert len(loaded) == 2
    assert loaded[0]["abstained"] is True
    assert loaded[1]["correct"] is True


def test_config_run_id_deterministic():
    from aq.config import ConfidenceConfig, DataConfig, ModelConfig, QuantConfig, RunConfig

    def mk():
        return RunConfig(
            model=ModelConfig(name="llama32_1b", hf_id="meta/llama", family="llama", params_b=1.0),
            quant=QuantConfig(method="gptq", bits=4),
            data=DataConfig(name="abstention_bench"),
            confidence=ConfidenceConfig(),
        )
    a, b = mk().run_id(), mk().run_id()
    assert a == b
    assert "gptq4" in a and "abstention_bench" in a


def test_real_model_smoke():
    """Tiny-model end-to-end smoke; skips if transformers/torch unavailable."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    # Stage 2: load sshleifer/tiny-gpt2, run 5 samples through run_inference,
    # assert raw record fields are complete. Intentionally not implemented until
    # the ML stack lands (Python-version-compatible env).
    pytest.skip("Stage 2: implement tiny-gpt2 end-to-end smoke")
