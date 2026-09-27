"""AbstentionBench loader (Kirichenko et al., 2025) -> list[Sample].

AbstentionBench is a *script-based* HF dataset that only works with
``datasets <= 3.6.0`` and pulls 20 upstream sources. To keep that fragility out
of the inference env (which runs datasets 4.x), we DO NOT call load_dataset
here. Instead a one-off extraction (scripts/extract_abstention_bench.py, run in
a datasets==3.6.0 env) materializes a subset to:

    data/processed/abstention_bench.jsonl

with one JSON object per line in the canonical schema:

    {"question": str, "reference_answers": list[str] | null,
     "should_abstain": bool, "category": str, "metadata_json": str}

This loader just reads that file -> Sample (answerable = not should_abstain).
"""

from __future__ import annotations

import json
from pathlib import Path

from ..config import DataConfig
from .base import Sample

DEFAULT_PATH = Path("data/processed/abstention_bench.jsonl")


def load(cfg: DataConfig) -> list[Sample]:
    path = Path(cfg.subset) if cfg.subset else DEFAULT_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run the one-off extraction first:\n"
            f"  python scripts/extract_abstention_bench.py --out {path} "
            f"--max-samples {cfg.max_samples or 1000}\n"
            f"(in a venv with datasets==3.6.0; see the script header)."
        )
    samples: list[Sample] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            should_abstain = bool(rec["should_abstain"])
            refs = rec.get("reference_answers") or []
            samples.append(Sample(
                id=rec.get("id", f"ab-{i}"),
                dataset="abstention_bench",
                question=rec["question"],
                answerable=not should_abstain,
                gold_answers=list(refs),
                category=rec.get("category", "unanswerable" if should_abstain else "answerable"),
                meta={"metadata_json": rec.get("metadata_json", "")},
            ))
    if cfg.max_samples is not None:
        samples = samples[: cfg.max_samples]
    return samples
