"""GGUF / llama.cpp inference backend (the second quantization family).

GGUF k-quants (Q8/Q6/Q5/Q4/Q3/Q2) via llama-cpp-python are the most
edge-realistic compression and run on any CPU/GPU (no Blackwell/sm_120 issues
that block GPTQ/AWQ). Used as the second quantization axis alongside
bitsandbytes.

Heavy deps (llama_cpp, huggingface_hub) imported lazily. Prompts are built with
the base model's HF chat template (as a string) for parity with the HF backend.
"""

from __future__ import annotations

from typing import Any


def load_gguf(repo_id: str, filename: str, *, n_ctx: int = 4096, n_gpu_layers: int = 0) -> Any:
    """Download (cached) a GGUF file from HF and open it with llama.cpp.

    n_gpu_layers=0 -> CPU (robust default); raise to offload layers to GPU when
    a CUDA-enabled llama.cpp build is available.

    NOTE on n_ctx: generate_gguf uses logits_all=True (required for per-token
    logprobs), which allocates an n_ctx x vocab logits buffer. For large-vocab
    models (e.g. Qwen ~152k) this is ~2.3 GB at n_ctx 4096 and ~4.6 GB at 8192;
    keep n_ctx modest to avoid OOM on a 16 GB box. Longer prompts are truncated.
    """
    from huggingface_hub import hf_hub_download  # noqa: PLC0415
    from llama_cpp import Llama  # noqa: PLC0415

    path = hf_hub_download(repo_id, filename)
    return Llama(
        model_path=path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        logits_all=True,   # required for per-token logprobs in create_completion
        verbose=False,
    )


def generate_gguf(llm: Any, prompt: str, *, max_new_tokens: int) -> tuple[str, list[float]]:
    """Greedy completion with per-token logprobs.

    Returns (text, token_logprobs). token_logprobs are natural-log probs of the
    sampled tokens (None entries filtered out).
    """
    # Guard against context overflow: truncate the prompt to leave room for
    # generation (llama.cpp errors if prompt+max_tokens exceed n_ctx).
    budget = llm.n_ctx() - max_new_tokens - 8
    ids = llm.tokenize(prompt.encode("utf-8"))
    if budget > 0 and len(ids) > budget:
        # keep head (system prompt) + tail (question + generation prompt), drop
        # the middle (long context); mirrors the HF backend's truncation.
        keep_head = min(384, budget // 3)
        ids = ids[:keep_head] + ids[-(budget - keep_head):]
        prompt = llm.detokenize(ids).decode("utf-8", errors="ignore")
    out = llm(
        prompt,
        max_tokens=max_new_tokens,
        temperature=0.0,
        logprobs=1,
        echo=False,
    )
    choice = out["choices"][0]
    text = choice["text"].strip()
    lp_info = choice.get("logprobs") or {}
    # coerce to native python float (llama.cpp may hand back numpy float32)
    token_logprobs = [float(x) for x in (lp_info.get("token_logprobs") or []) if x is not None]
    return text, token_logprobs
