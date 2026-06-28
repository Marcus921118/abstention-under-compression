"""Model loading: FP16 baseline and bitsandbytes-quantized variants.

Heavy deps (torch/transformers) are imported lazily so the package still
imports on the metrics-only env (Python 3.14). GPTQ/AWQ live in
``aq.models.quantize`` and arrive in Stage 3; G2 uses FP16 + bnb (NF4 / int8),
which quantize at load time and need no calibration set.

Contract (workflow §0.6): exactly one model resident in VRAM at a time; the
caller releases it via ``release_model``.
"""

from __future__ import annotations

from typing import Any

from ..config import RunConfig


def _resolve_dtype(name: str):
    import torch  # noqa: PLC0415

    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}.get(name, torch.float16)


def load_model(cfg: RunConfig) -> tuple[Any, Any]:
    """Load (model, tokenizer) for the FP16 or bnb-quantized variant in cfg.

    Supported in G2: quant.method in {"fp16", "bnb_nf4"}.
      - fp16            -> dtype float16 on CUDA
      - bnb_nf4 bits=4  -> 4-bit NF4 (double-quant per extra)
      - bnb_nf4 bits=8  -> 8-bit int8
    GPTQ/AWQ raise NotImplementedError until Stage 3.
    """
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    hf_id = cfg.model.hf_id
    method = cfg.quant.method

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if method == "fp16":
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, dtype=torch.float16, device_map="cuda"
        )
    elif method == "bnb_nf4":
        from transformers import BitsAndBytesConfig  # noqa: PLC0415

        compute_dtype = _resolve_dtype(cfg.quant.extra.get("compute_dtype", "bfloat16"))
        if cfg.quant.bits == 4:
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=bool(cfg.quant.extra.get("double_quant", True)),
            )
        elif cfg.quant.bits == 8:
            bnb = BitsAndBytesConfig(load_in_8bit=True)
        else:
            raise ValueError(f"bnb_nf4 supports bits in {{4,8}}, got {cfg.quant.bits}")
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, quantization_config=bnb, device_map={"": 0}
        )
    elif method in ("gptq", "awq"):
        raise NotImplementedError(f"{method}: Stage 3 (aq.models.quantize)")
    else:
        raise ValueError(f"unknown quant method: {method}")

    model.eval()
    return model, tokenizer


def release_model(model: Any) -> None:
    """Free a model from VRAM (del + empty_cache). Safe if torch absent."""
    try:
        import gc

        import torch  # noqa: PLC0415

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
