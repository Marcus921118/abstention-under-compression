"""SQuAD 2.0 loader -> list[Sample] (open, no-auth abstention task).

SQuAD 2.0 (rajpurkar/squad_v2) pairs each question with a context paragraph.
~1/3 of questions are *unanswerable from the context* (is_impossible). That maps
cleanly onto abstention: when the answer is not in the provided context, the
correct behaviour is to abstain ("insufficient information"). Loads directly
with datasets 4.x -- no script, no gated sources, no gdown.

Mapping:
  answerable   = len(answers['text']) > 0
  gold_answers = unique answers['text']
  category     = "answerable" | "no_answer_in_context"
  question     = "Context: <context>\n\nQuestion: <question>"
"""

from __future__ import annotations

import random

from ..config import DataConfig
from .base import Sample

HF_ID = "rajpurkar/squad_v2"


def load(cfg: DataConfig) -> list[Sample]:
    from datasets import load_dataset  # noqa: PLC0415

    split = cfg.split if cfg.split in ("train", "validation") else "validation"
    ds = load_dataset(HF_ID, split=split)

    answerable_rows, unanswerable_rows = [], []
    for ex in ds:
        ans = list(dict.fromkeys(ex["answers"]["text"]))  # unique, order-preserving
        is_answerable = len(ans) > 0
        question = f"Context: {ex['context']}\n\nQuestion: {ex['question']}"
        s = Sample(
            id=str(ex["id"]),
            dataset="squad2",
            question=question,
            answerable=is_answerable,
            gold_answers=ans,
            category="answerable" if is_answerable else "no_answer_in_context",
        )
        (answerable_rows if is_answerable else unanswerable_rows).append(s)

    # Balanced subset for a manageable, interpretable run.
    rng = random.Random(cfg.seed)
    rng.shuffle(answerable_rows)
    rng.shuffle(unanswerable_rows)
    if cfg.max_samples is not None:
        half = cfg.max_samples // 2
        samples = answerable_rows[:half] + unanswerable_rows[: cfg.max_samples - half]
        rng.shuffle(samples)
    else:
        samples = answerable_rows + unanswerable_rows
        rng.shuffle(samples)
    return samples
