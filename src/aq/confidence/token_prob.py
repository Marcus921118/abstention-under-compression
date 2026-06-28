"""Token / sequence probability confidence (logit-based).

STUB for Gate G1. The aggregation math (mean token logprob -> confidence) is
pure and unit-testable; extraction of logprobs from a model is Stage 2.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def sequence_confidence_from_logprobs(token_logprobs: Sequence[float]) -> float | None:
    """Geometric-mean token probability of the answer span, in [0,1].

    confidence = exp(mean(token_logprobs)). Returns None for empty input.
    Pure function -- unit-testable without a model.
    """
    lps = [lp for lp in token_logprobs if lp is not None]
    if not lps:
        return None
    return float(math.exp(sum(lps) / len(lps)))
