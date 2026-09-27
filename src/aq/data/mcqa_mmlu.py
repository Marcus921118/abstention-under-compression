"""Judge-free MCQA abstention task built from MMLU (open, no auth).

Construction (balanced, controllable, judge-free):
  * answerable item: 3 content options = the correct choice + 2 distractors
    (shuffled into A-C) plus "D) Cannot be determined from the options". The gold
    answer is the content letter; choosing D is over-abstention.
  * unanswerable item: 3 content options = 3 distractors with the CORRECT choice
    REMOVED (so the answer is genuinely absent), plus "D) Cannot be determined".
    The gold answer is D — the model should abstain.

Scoring is exact option-letter matching (aq.eval.mcqa), so no LLM judge is
needed and degenerate low-bit text cannot fool a text judge. Single-letter
answers also remove long-generation error-propagation from the abstention signal.
"""

from __future__ import annotations

import random

from ..config import DataConfig
from .base import Sample

HF_ID = "cais/mmlu"
ABSTAIN_LETTER = "D"
ABSTAIN_TEXT = "Cannot be determined from the options"
_LETTERS = ["A", "B", "C", "D"]


def _format(question: str, options: list[str]) -> str:
    body = "\n".join(f"{L}) {opt}" for L, opt in zip(_LETTERS, options, strict=False))
    return (
        f"{question}\n\n{body}\n\n"
        "Answer with the single letter of the best option."
    )


def load(cfg: DataConfig) -> list[Sample]:
    from datasets import load_dataset  # noqa: PLC0415

    subset = cfg.subset or "all"
    split = cfg.split if cfg.split in ("test", "validation", "dev") else "test"
    ds = load_dataset(HF_ID, subset, split=split)

    n = cfg.max_samples if cfg.max_samples is not None else len(ds)
    # 2026-07-26 FIX: this was `random.Random(0)`, hardcoded, so every data seed
    # drew the IDENTICAL item subset. With greedy decoding that made all "seeds"
    # byte-identical runs -- a 5-seed study with 1 independent observation.
    # Caught only because the paired t reported t=inf (zero variance).
    # squad2.py:48 always did this correctly; mcqa_mmlu did not.
    rng = random.Random(cfg.seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    idxs = idxs[:n]

    samples: list[Sample] = []
    for i, di in enumerate(idxs):
        ex = ds[di]
        choices = list(ex["choices"])
        ans = int(ex["answer"])
        if len(choices) < 4:
            continue
        correct_text = choices[ans]
        distractors = [c for j, c in enumerate(choices) if j != ans]
        make_unanswerable = (i % 2 == 1)
        if make_unanswerable:
            # correct removed: 3 distractors as content options -> abstain is gold
            content = distractors[:3]
            rng.shuffle(content)
            options = content + [ABSTAIN_TEXT]
            gold_letter = ABSTAIN_LETTER
            answerable = False
            category = "no_correct_option"
        else:
            content = [correct_text] + distractors[:2]
            rng.shuffle(content)
            options = content + [ABSTAIN_TEXT]
            gold_letter = _LETTERS[options.index(correct_text)]
            answerable = True
            category = "answerable"
        samples.append(Sample(
            id=f"mcqa-{di}",
            dataset="mcqa_mmlu",
            question=_format(ex["question"], options),
            answerable=answerable,
            gold_answers=[gold_letter],
            category=category,
            meta={"mcqa": True, "gold_letter": gold_letter,
                  "abstain_letter": ABSTAIN_LETTER,
                  "abstain_option_tokens": [ABSTAIN_LETTER, ABSTAIN_TEXT]},
        ))
    rng.shuffle(samples)
    return samples
