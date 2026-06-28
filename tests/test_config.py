"""Config layering + run_id tests using the real YAML fragments."""

from pathlib import Path

import pytest

from aq.config import QuantConfig, load_run_config

CFG = Path(__file__).resolve().parents[1] / "configs"


def test_load_run_config_layers_yaml():
    cfg = load_run_config(
        base=CFG / "base.yaml",
        model=CFG / "models" / "llama32_1b.yaml",
        quant=CFG / "quant" / "gptq.yaml",
        data={"name": "abstention_bench", "split": "test", "max_samples": 100},
    )
    assert cfg.model.hf_id == "meta-llama/Llama-3.2-1B-Instruct"
    assert cfg.quant.method == "gptq" and cfg.quant.bits == 4
    assert cfg.seed == 0 and cfg.batch_size == 8
    assert "verbalized" in cfg.confidence.methods


def test_override_bits():
    cfg = load_run_config(
        base=CFG / "base.yaml",
        model=CFG / "models" / "qwen25_15b.yaml",
        quant=CFG / "quant" / "gptq.yaml",
        data={"name": "triviaqa"},
        overrides={"quant": {"bits": 3}},
    )
    assert cfg.quant.bits == 3
    assert "gptq3" in cfg.run_id()


def test_invalid_bits_rejected():
    with pytest.raises(ValueError):
        QuantConfig(method="gptq", bits=1)   # below supported range (2..16)
    with pytest.raises(ValueError):
        QuantConfig(method="gptq", bits=32)  # above supported range


def test_raw_path_uses_results_dir():
    cfg = load_run_config(
        base=CFG / "base.yaml",
        model=CFG / "models" / "phi3_mini.yaml",
        quant=CFG / "quant" / "fp16.yaml",
        data={"name": "medabstain"},
    )
    assert cfg.raw_path().parts[-2] == "raw"
    assert cfg.raw_path().suffix == ".jsonl"
