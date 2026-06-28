"""Semantic entropy (Farquhar et al., 2024) -- optional, most expensive signal.

STUB for Gate G1 / Stage 4 (workflow §0.6, §7). Requires N samples per question
+ an NLI bidirectional-entailment clustering step. Listed last.
"""

from __future__ import annotations

from collections.abc import Sequence


def semantic_entropy(samples: Sequence[str], entail_fn=None) -> float:
    """Cluster sampled answers by meaning, return entropy over clusters."""
    raise NotImplementedError("Stage 4: semantic entropy via NLI clustering")
