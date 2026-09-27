"""MedAbstain loader (medical MCQA, high-risk) -> list[Sample].

STUB for Gate G1. Stage 2 maps each MCQA item to a Sample; unanswerable items
carry an explicit abstain option recorded in meta["abstain_option_tokens"].
"""

from __future__ import annotations

from ..config import DataConfig
from .base import Sample


def load(cfg: DataConfig) -> list[Sample]:
    raise NotImplementedError("Stage 2: load MedAbstain (verify license first)")
