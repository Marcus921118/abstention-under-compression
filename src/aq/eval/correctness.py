"""Answer-correctness scoring for answerable questions (pure, testable).

G2 uses normalized substring matching against any gold answer -- adequate for
short-answer factual QA. Stage 3 adds MCQA option matching and dataset-specific
normalizers. Kept separate from metrics so the matching policy is swappable.
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
    """True if any gold answer appears as a normalized token-substring of output.

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
        # word-boundary-aware containment on the normalized strings
        if re.search(rf"(?:^| ){re.escape(ng)}(?: |$)", norm_out):
            return True
    return False
