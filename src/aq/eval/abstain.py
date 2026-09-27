"""Detect whether a model output is an abstention ("I don't know" / refusal).

Pure, model-free, and unit-testable. This is the operationalization of the
study's primary dependent variable, so it is kept deliberately simple and
transparent; Stage 2 may add dataset-specific patterns (e.g. an explicit
"Option E: cannot be determined" for MedAbstain MCQA).

Unified 2026-07-20 (P0-1, STATE.md future-plan) with the h4 pipeline's
``experiments/scoring_v2.is_refusal``: the two classifiers diverged on real
h4_raw data (8/176 phi3 parametric records; see
``results/scorer_unification_impact.json``), which made cross-pipeline
comparisons invalid (PITFALLS.md Sec5). The merge keeps this module's broader
regex coverage (v2 was pure keyword substring matching) but adopts v2's
composite AND guard as an *opt-in*: pass ``gold_answers`` to also require that
the fixed-up correctness check finds no gold answer in the text, so a hedge
like "I'm not sure, but I believe it is Paris" (gold=["Paris"]) is not
misjudged as abstention. Callers that omit ``gold_answers`` keep the old
keyword-only behavior (this is also v2's own designed fallback for items with
no gold, e.g. genuinely unanswerable ones).
"""

from __future__ import annotations

import re

from .correctness import is_correct

# Phrases that signal the model declined to commit to an answer.
# Curated to match the abstention the SYSTEM_PROMPT requests ("I don't know.")
# plus common hedges/refusals. NOTE: this regex detector is the G2/G3 baseline;
# an LLM-judge (as in the AbstentionBench paper) is the Stage-4 upgrade for the
# final abstention labels.
_ABSTAIN_PATTERNS = [
    r"\bi (?:do not|don't|cannot|can't|can not) know\b",
    r"\bi (?:do not|don't) have (?:enough|sufficient|the|an exact|access)",
    r"\bi (?:cannot|can't|can not|am unable to|am not able to) "
    r"(?:answer|determine|tell|say|know|provide|confirm|verify|predict)\b",
    r"\b(?:cannot|can't|can not|impossible to|not possible to) "
    r"be (?:answered|determined|known|predicted)\b",
    r"\b(?:cannot|can't|can not|impossible|not possible) to "
    r"(?:answer|determine|know|predict|say)\b",
    r"\b(?:there is |there's )?(?:not enough|insufficient|no) "
    r"(?:information|context|data|way to know)\b",
    r"\b(?:would|i'd) need more (?:information|context|details)\b",
    r"\bunanswerable\b",
    r"\bno (?:correct |single |definitive |one )?answer\b",
    r"\b(?:unable|not able) to provide\b",
    r"\bi'?m (?:not sure|unsure|uncertain)\b",
    r"\bconsult (?:a|your) (?:doctor|physician|professional|specialist)\b",
    r"\b(?:depends on|without knowing|without more) \b",
    # merged in from scoring_v2.REFUSAL_KEYWORDS (2026-07-20 unification):
    # bare "cannot answer" (no leading "i "), and hedges v2 had but this
    # module's patterns did not.
    r"\bcannot answer\b",
    r"\bno idea\b",
    r"\bnot mentioned\b",
    # CJK keywords straight from scoring_v2.REFUSAL_KEYWORDS -- literal
    # substrings, no \b (word-boundary is a \w-based ASCII concept and does
    # not segment CJK text usefully).
    r"我不知道", r"無法回答", r"無法提供", r"資訊不足", r"沒有提及",
    r"無法確定", r"文中未提及",
]

_ABSTAIN_RE = re.compile("|".join(_ABSTAIN_PATTERNS), re.IGNORECASE)


def is_abstention(
    text: str,
    *,
    abstain_option_tokens: list[str] | None = None,
    gold_answers: list[str] | None = None,
) -> bool:
    """Return True if the output reads as an abstention / refusal.

    ``abstain_option_tokens``: for MCQA tasks, exact option markers that mean
    "abstain" (e.g. ["E", "cannot be determined"]); matched case-insensitively.
    An option-token hit is a structural signal (the model literally picked the
    abstain choice), not a hedge phrase, so it is decided independently of the
    ``gold_answers`` guard below.

    ``gold_answers``: optional gold answers for the composite v2-style guard --
    when given and non-empty, a keyword/pattern hit is only counted as
    abstention if the (unified) correctness check also finds no gold answer in
    the text. Omit (or pass ``None``/``[]``) to fall back to keyword-only,
    matching the pre-unification behavior of this function.
    """
    if not text:
        return False
    if abstain_option_tokens:
        low = text.lower()
        for tok in abstain_option_tokens:
            if tok.lower() in low:
                return True
    if not _ABSTAIN_RE.search(text):
        return False
    if gold_answers:
        return not is_correct(text, gold_answers)
    return True
