"""Answer-correctness scoring for answerable questions (pure, testable).

G2 uses normalized substring matching against any gold answer -- adequate for
short-answer factual QA. Stage 3 adds MCQA option matching and dataset-specific
normalizers. Kept separate from metrics so the matching policy is swappable.

Unified 2026-07-20 (P0-1, STATE.md future-plan) with the h4 pipeline's
``experiments/scoring_v2.check_accuracy``: this module's containment was
one-directional (gold contained in output only), while v2 also matched the
reverse direction (output contained in gold) -- needed for short correct
answers like "yes"/"no" nested inside a longer gold string. The unified
``is_correct`` keeps this module's NFKD normalization (handles accents; more
general than v2's ASCII-punctuation strip) and adds v2's second direction. See
``results/scorer_unification_impact.json`` for the measured impact (it was
small: 1/176 records on real parametric data).
"""

from __future__ import annotations

import re
import unicodedata

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")


def normalize(text: str) -> str:
    """Lowercase, strip accents/punctuation/articles, collapse whitespace."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _ARTICLES_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def is_correct(output_text: str, gold_answers: list[str]) -> bool:
    """True if any gold answer and the output share a token-boundary containment.

    Bidirectional (matches scoring_v2.check_accuracy): either the gold answer
    is contained in the output, or (for a short correct output like "yes")
    the output is contained in the gold answer.

    Returns False when there are no gold answers (e.g. unanswerable items).
    """
    if not gold_answers:
        return False
    norm_out = normalize(output_text)
    if not norm_out:
        return False
    for gold in gold_answers:
        ng = normalize(gold)
        if not ng:
            continue
        # word-boundary-aware containment on the normalized strings, both
        # directions
        if re.search(rf"(?:^| ){re.escape(ng)}(?: |$)", norm_out):
            return True
        if re.search(rf"(?:^| ){re.escape(norm_out)}(?: |$)", ng):
            return True
    return False
