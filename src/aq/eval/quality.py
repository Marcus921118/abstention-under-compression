"""Output-quality controls that separate genuine over-confidence from
instruction-following / capability collapse (pure, model-free, testable).

These address two reviewer confounds:
  (1) "Did the quantized model abstain less because it is over-confident, or
       because quantization broke its ability to follow the abstain
       instruction / produce coherent text?" -> format compliance + degeneration.
  (2) "Is the monitoring-vs-behaviour decoupling just autoregressive error
       propagation in long generations?" -> output length, so decoupling can be
       reported on length-matched, non-degenerate outputs.

All functions operate on the generated text already stored in raw records, so
they can be computed retroactively without re-running any model.
"""

from __future__ import annotations

import re
from collections import Counter

# Same pattern the verbalized-confidence parser looks for; presence means the
# model still followed the output-format instruction.
_CONF_FORMAT_RE = re.compile(r"confidence\s*[:=]?\s*[0-9]{1,3}(?:\.[0-9]+)?\s*%", re.IGNORECASE)
_CHAR_RUN_RE = re.compile(r"(.)\1{19,}")  # 20+ identical chars in a row


def follows_confidence_format(text: str) -> bool:
    """True if the output contains the requested 'Confidence: N%' format."""
    return bool(text) and bool(_CONF_FORMAT_RE.search(text))


def n_words(text: str) -> int:
    return len((text or "").split())


def is_degenerate(text: str) -> bool:
    """Heuristic for broken/degenerate output (a capability-collapse signature).

    Flags: empty/near-empty; a long run of one character; or low lexical
    diversity in a non-trivial output (heavy token repetition).
    """
    t = (text or "").strip()
    if len(t) < 2:
        return True
    if _CHAR_RUN_RE.search(t):
        return True
    words = t.split()
    if len(words) >= 8:
        diversity = len(set(words)) / len(words)
        if diversity < 0.35:
            return True
        # any single token dominating the output
        most = Counter(words).most_common(1)[0][1]
        if most / len(words) > 0.5:
            return True
    return False


def quality_flags(text: str) -> dict:
    """Bundle used by the pipeline / retroactive control analysis."""
    return {
        "format_ok": follows_confidence_format(text),
        "degenerate": is_degenerate(text),
        "n_words": n_words(text),
    }
