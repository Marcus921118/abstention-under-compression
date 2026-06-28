"""Quantization wrappers: GPTQ / AWQ / bitsandbytes-NF4 (+ optional GGUF).

STUB for Gate G1. Implement in Stage 2 (workflow §3 "量化封裝").
Each function takes an FP16 model handle / hf_id + QuantConfig and returns an
inference-ready quantized model. Cache quantized weights by run_id.
"""

from __future__ import annotations

from typing import Any

from ..config import QuantConfig


def quantize_gptq(hf_id: str, q: QuantConfig) -> Any:
    raise NotImplementedError("Stage 2: auto-gptq wrapper")


def quantize_awq(hf_id: str, q: QuantConfig) -> Any:
    raise NotImplementedError("Stage 2: autoawq wrapper")


def quantize_bnb_nf4(hf_id: str, q: QuantConfig) -> Any:
    raise NotImplementedError("Stage 2: bitsandbytes NF4 wrapper")
