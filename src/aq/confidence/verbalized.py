"""Verbalized confidence: ask the model for a 0-100% certainty.

STUB for Gate G1. The prompt template and parser live here; the parser is pure
and unit-testable independently of any model.
"""

from __future__ import annotations

import re

# Appended to the task prompt to elicit a self-reported confidence.
CONFIDENCE_SUFFIX = (
    "\n\nAfter your answer, state your confidence that the answer is correct "
    "as a percentage from 0 to 100, on a line formatted exactly as: "
    "Confidence: <number>%"
)

_CONF_RE = re.compile(r"confidence\s*[:=]?\s*([0-9]{1,3}(?:\.[0-9]+)?)\s*%", re.IGNORECASE)


def parse_verbalized_confidence(text: str) -> float | None:
    """Extract a [0,1] confidence from model output, or None if absent.

    Pure function -- safe to unit-test without a model.
    """
    if not text:
        return None
    matches = _CONF_RE.findall(text)
    if not matches:
        return None
    val = float(matches[-1])  # take the last stated confidence
    val = max(0.0, min(100.0, val))
    return val / 100.0
