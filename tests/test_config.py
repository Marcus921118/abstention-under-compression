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


def test_data_seed_defaults_zero_and_affects_run_id():
    """data.seed (independent-sample support for the real-quant decisive
    test's seeds>=3 bolstering) defaults to 0 for backward compat, and
    distinct data seeds must not collide on run_id/cache path."""
    cfg0 = load_run_config(
        base=CFG / "base.yaml",
        model=CFG / "models" / "qwen25_15b.yaml",
        quant=CFG / "quant" / "fp16.yaml",
        data={"name": "squad2", "max_samples": 160},
    )
    assert cfg0.data.seed == 0
    cfg1 = load_run_config(
        base=CFG / "base.yaml",
        model=CFG / "models" / "qwen25_15b.yaml",
        quant=CFG / "quant" / "fp16.yaml",
        data={"name": "squad2", "max_samples": 160, "seed": 1},
    )
    assert cfg1.data.seed == 1
    assert cfg0.run_id() != cfg1.run_id()


def test_skip_layers_set_affects_run_id():
    """R3 matched arm (docs/實驗計畫_R系列_20260718.md): distinct protected-layer
    sets must not collide on run_id, or cache-first would silently reuse the
    wrong arm's raw jsonl (DIAGNOSIS.md failure mode 2)."""

    def mk(skip_layers):
        return load_run_config(
            base=CFG / "base.yaml", model=CFG / "models" / "qwen25_15b.yaml",
            quant=CFG / "quant" / "bnb_nf4.yaml",
            data={"name": "squad2", "max_samples": 160},
            overrides={"quant": {"bits": 4, "extra": {"skip_layers": skip_layers}}},
        )

    calib_id = mk([0, 15, 18]).run_id()
    matched_id = mk([4, 14, 15]).run_id()
    assert calib_id != matched_id
    existing_raw_hashes = {p.stem.rsplit("__", 1)[-1] for p in (Path(__file__).resolve().parents[1] / "results" / "raw").glob("*.jsonl")}
    assert matched_id.rsplit("__", 1)[-1] not in existing_raw_hashes


def test_raw_path_uses_results_dir():
    cfg = load_run_config(
        base=CFG / "base.yaml",
        model=CFG / "models" / "phi3_mini.yaml",
        quant=CFG / "quant" / "fp16.yaml",
        data={"name": "medabstain"},
    )
    assert cfg.raw_path().parts[-2] == "raw"
    assert cfg.raw_path().suffix == ".jsonl"
