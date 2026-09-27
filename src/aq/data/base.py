"""Unified sample schema shared by every dataset loader.

Each benchmark loader (AbstentionBench / MedAbstain / TriviaQA) normalizes its
records into ``Sample`` so the inference pipeline and metrics never special-case
a dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Sample:
    """One evaluation item, dataset-agnostic.

    Attributes
    ----------
    id:           stable unique id within the dataset.
    dataset:      source dataset name (e.g. "abstention_bench").
    question:     the prompt text shown to the model.
    answerable:   True if a correct answer exists; False = should-abstain.
    gold_answers: acceptable answer strings (empty for unanswerable items).
    category:     fine-grained reason for unanswerability (e.g. "false_premise",
                  "underspecified", "stale", or "answerable").
    meta:         free-form extra fields kept for traceability.
    """

    id: str
    dataset: str
    question: str
    answerable: bool
    gold_answers: list[str] = field(default_factory=list)
    category: str = "answerable"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GenerationRecord:
    """One raw record written per (sample, run) to results/raw/{run_id}.jsonl.

    Generation and scoring are decoupled (workflow §0.3): metrics are computed
    later from these records, so re-scoring never re-runs the model.
    """

    run_id: str
    sample_id: str
    dataset: str
    answerable: bool
    category: str
    output_text: str
    abstained: bool
    correct: bool | None            # None when not applicable (e.g. unanswerable)
    verbalized_confidence: float | None = None
    token_logprob: float | None = None
    semantic_entropy: float | None = None
    samples: list[str] = field(default_factory=list)  # for semantic entropy
    meta: dict[str, Any] = field(default_factory=dict)
