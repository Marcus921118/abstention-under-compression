"""Judge-free multiple-choice scoring with an explicit abstain option.

This is the control task that defuses two reviewer concerns at once:
  * the answer is a single option letter, so abstention is read from one decision
    rather than a long free-text generation (removes autoregressive-error-
    propagation worries about the monitoring-vs-behaviour comparison); and
  * correctness and abstention are exact-letter matches, so NO LLM judge is
    needed and broken/degenerate low-bit output cannot fool a text judge.

An item presents content options plus one explicit "Cannot be determined from the
options" option (the abstain letter). For unanswerable items the abstain letter
is the gold answer; for answerable items a content letter is gold and choosing
the abstain letter is over-abstention.
"""

from __future__ import annotations

import re

# Match the model's chosen option: "B", "(B)", "B)", "Answer: B", "The answer is B".
_CHOICE_RE = re.compile(
    r"(?:answer\s*(?:is|:)?\s*)?\(?\b([A-Ea-e])\b\)?", re.IGNORECASE
)


def extract_choice(text: str, valid: str = "ABCDE") -> str | None:
    """Return the first option letter the output commits to, or None.

    Prefers an explicit 'answer is X' / 'X)' cue; falls back to the first
    standalone A-E letter. Returns an uppercase letter in `valid`, else None.
    """
    if not text:
        return None
    for m in _CHOICE_RE.finditer(text):
        letter = m.group(1).upper()
        if letter in valid:
            return letter
    return None


def mcqa_outcome(text: str, gold_letter: str, abstain_letter: str) -> dict:
    """Classify an MCQA output. Returns abstained/correct/choice (all judge-free).

    correct is None when the item is abstained or has no gold content answer.
    """
    from .abstain import is_abstention  # noqa: PLC0415

    choice = extract_choice(text)
    if choice == abstain_letter:
        return {"choice": choice, "abstained": True, "correct": None}
    if choice is None and is_abstention(text):
        return {"choice": None, "abstained": True, "correct": None}
    correct = None
    if gold_letter != abstain_letter:
        correct = choice is not None and choice == gold_letter
    return {"choice": choice, "abstained": False, "correct": correct}
