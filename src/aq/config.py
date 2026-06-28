"""Config-driven experiment definitions (workflow §0.2).

Every experiment is a YAML file; programs never hard-code model names,
bit-widths, or paths. A deterministic ``run_id`` is derived from the resolved
config so reruns hit the cache.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class ModelConfig(BaseModel):
    name: str                       # short handle, e.g. "llama32_1b"
    hf_id: str                      # huggingface repo id
    family: str = "unknown"
    params_b: float = 0.0           # parameter count in billions


class QuantConfig(BaseModel):
    method: Literal["fp16", "gptq", "awq", "bnb_nf4", "gguf"]
    bits: int = 16                  # nominal bit-width label (16,8,6,5,4,3,2)
    group_size: int | None = 128
    gguf_repo: str | None = None    # HF repo holding GGUF files (method=gguf)
    gguf_file: str | None = None    # specific .gguf filename for this bit-width
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("bits")
    @classmethod
    def _bits_ok(cls, v: int) -> int:
        if not (2 <= v <= 16):
            raise ValueError(f"unsupported bit-width: {v}")
        return v


class DataConfig(BaseModel):
    name: str                       # "abstention_bench" | "medabstain" | "triviaqa"
    split: str = "test"
    max_samples: int | None = None  # None = all
    subset: str | None = None


class ConfidenceConfig(BaseModel):
    methods: list[Literal["verbalized", "token_prob", "semantic_entropy"]] = Field(
        default_factory=lambda: ["verbalized", "token_prob"]
    )
    n_samples: int = 5              # for semantic entropy
    overconfidence_tau: float = 0.8


class RunConfig(BaseModel):
    """Fully-resolved single experiment (one cell of the core matrix)."""

    model: ModelConfig
    quant: QuantConfig
    data: DataConfig
    confidence: ConfidenceConfig = Field(default_factory=ConfidenceConfig)
    seed: int = 0
    batch_size: int = 8
    max_new_tokens: int = 256
    results_dir: str = "results"

    def run_id(self) -> str:
        """Deterministic hash over the salient fields."""
        payload = {
            "model": self.model.hf_id,
            "quant": {"m": self.quant.method, "b": self.quant.bits,
                      "g": self.quant.group_size, "e": self.quant.extra,
                      "gf": self.quant.gguf_file},
            "data": {"n": self.data.name, "s": self.data.split,
                     "max": self.data.max_samples, "sub": self.data.subset},
            "conf": sorted(self.confidence.methods),
            "seed": self.seed,
            "max_new_tokens": self.max_new_tokens,
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        h = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]
        return f"{self.model.name}__{self.quant.method}{self.quant.bits}__{self.data.name}__{h}"

    def raw_path(self) -> Path:
        return Path(self.results_dir) / "raw" / f"{self.run_id()}.jsonl"


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_run_config(
    *,
    base: str | Path,
    model: str | Path,
    quant: str | Path,
    data: dict[str, Any],
    overrides: dict[str, Any] | None = None,
) -> RunConfig:
    """Compose a RunConfig from layered YAML fragments + a data spec.

    Layering: base.yaml <- model/*.yaml <- quant/*.yaml <- data dict <- overrides.
    """
    merged: dict[str, Any] = load_yaml(base)
    merged = _deep_merge(merged, {"model": load_yaml(model)})
    merged = _deep_merge(merged, {"quant": load_yaml(quant)})
    merged = _deep_merge(merged, {"data": data})
    if overrides:
        merged = _deep_merge(merged, overrides)
    return RunConfig(**merged)
