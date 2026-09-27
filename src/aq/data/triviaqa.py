"""TriviaQA loader (answerable control) -> list[Sample].

STUB for Gate G1. Used to separate accuracy degradation from abstention-
behaviour degradation. All items answerable=True.
"""

from __future__ import annotations

from ..config import DataConfig
from .base import Sample


def load(cfg: DataConfig) -> list[Sample]:
    raise NotImplementedError("Stage 2: load TriviaQA answerable control")
