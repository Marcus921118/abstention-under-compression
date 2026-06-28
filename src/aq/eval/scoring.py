"""Score raw generation records into per-run metrics (workflow §4, decoupled).

Reads results/raw/{run_id}.jsonl and produces the metric dict from
aq.eval.metrics.compute_run_metrics. The confidence *signal* is selectable so
the same records can be scored under token-prob or verbalized confidence
without re-running the model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .metrics import compute_run_metrics
from .pipeline import read_records


def _confidence(rec: dict, signal: str) -> float:
    """Pick a [0,1] confidence for a record under the chosen signal.

    Falls back to the other signal, then to 0.5 (max ignorance), so every
    answered item contributes to calibration metrics.
    """
    tok = (rec.get("meta") or {}).get("token_prob_confidence")
    verb = rec.get("verbalized_confidence")
    primary, secondary = (verb, tok) if signal == "verbalized" else (tok, verb)
    for v in (primary, secondary):
        if v is not None:
            return float(v)
    return 0.5


def extract_arrays(records: list[dict], signal: str = "token_prob"):
    answerable = np.array([bool(r["answerable"]) for r in records])
    abstained = np.array([bool(r["abstained"]) for r in records])
    correct = np.array([bool(r.get("correct")) for r in records])  # None -> False
    confidence = np.array([_confidence(r, signal) for r in records])
    return answerable, abstained, correct, confidence


def score_run(
    path: str | Path,
    *,
    signal: str = "token_prob",
    tau: float = 0.8,
) -> dict[str, float]:
    """Compute the full per-run metric set from a raw JSONL file."""
    records = read_records(path)
    answerable, abstained, correct, confidence = extract_arrays(records, signal)
    metrics = compute_run_metrics(answerable, abstained, correct, confidence, tau=tau)
    metrics["signal"] = signal
    metrics["path"] = str(path)
    return metrics
