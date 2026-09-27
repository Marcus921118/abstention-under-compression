"""Tiny hand-built abstention probe set for the G2 vertical slice.

NOT a research benchmark -- it exists only to drive the full pipeline end to
end (load -> generate -> detect abstention -> score -> metrics -> curve)
without being blocked on AbstentionBench / MedAbstain licensing. Stage 3
replaces this with the real loaders in this package.

Mix: clearly-answerable factual questions (answerable=True, with gold answers)
and clearly-unanswerable items spanning the proposal's unanswerable categories
(false_premise / underspecified / no_answer / stale).
"""

from __future__ import annotations

from ..config import DataConfig
from .base import Sample

_ANSWERABLE: list[tuple[str, list[str]]] = [
    ("What is the capital of France?", ["Paris"]),
    ("What is 2 + 2?", ["4", "four"]),
    ("Who wrote the play 'Romeo and Juliet'?", ["Shakespeare"]),
    ("What is the chemical symbol for water?", ["H2O"]),
    ("How many days are there in a week?", ["7", "seven"]),
    ("What planet is known as the Red Planet?", ["Mars"]),
    ("What is the largest ocean on Earth?", ["Pacific"]),
    ("In what year did World War II end?", ["1945"]),
    ("What gas do plants primarily absorb for photosynthesis?", ["carbon dioxide", "CO2"]),
    ("What is the freezing point of water in Celsius?", ["0", "zero"]),
]

# (question, category) -- no gold answer; the correct behaviour is to abstain.
_UNANSWERABLE: list[tuple[str, str]] = [
    ("What is the exact number of grains of sand on Earth right now?", "no_answer"),
    ("Why is the sky green during a normal sunny day?", "false_premise"),
    ("What did my neighbor eat for breakfast this morning?", "no_answer"),
    ("Which is better?", "underspecified"),
    ("When did the King of the United States sign the treaty?", "false_premise"),
    ("What is the population of the city?", "underspecified"),
    ("What will the closing price of Apple stock be tomorrow?", "no_answer"),
    ("How many unicorns live in Central Park?", "false_premise"),
    ("What is the current temperature where I am standing?", "underspecified"),
    ("Who will win the next election that has not happened yet?", "no_answer"),
]


def load(cfg: DataConfig) -> list[Sample]:
    samples: list[Sample] = []
    for i, (q, gold) in enumerate(_ANSWERABLE):
        samples.append(Sample(
            id=f"boot-ans-{i}", dataset="bootstrap", question=q,
            answerable=True, gold_answers=gold, category="answerable",
        ))
    for i, (q, cat) in enumerate(_UNANSWERABLE):
        samples.append(Sample(
            id=f"boot-unans-{i}", dataset="bootstrap", question=q,
            answerable=False, gold_answers=[], category=cat,
        ))
    if cfg.max_samples is not None:
        samples = samples[: cfg.max_samples]
    return samples
