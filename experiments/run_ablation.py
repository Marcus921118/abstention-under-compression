#!/usr/bin/env python3
"""Ablation experiments: calibration data (A) and rounding/clipping decomposition (B).

Experiment A — Calibration Data Ablation:
  Does MCQA-style calibration data prevent ar1000 overfit?
  Conditions: {ar200, ar1000} × {pile-10k (done), mcqa (new)} + fp16 baseline.

Experiment B — Rounding vs Clipping Decomposition:
  Which AutoRound component causes the non-monotonic AUROC?
  Conditions: ar200 × {both (done), rounding_only (new)} + fp16 baseline.

Pre-requisites:
    # Pre-quantize new models on GPU server:
    CUDA_VISIBLE_DEVICES=3 python3 -u experiments/prequantize_ablation.py --all

Usage:
    # Generation (GPU server):
    CUDA_VISIBLE_DEVICES=3 python3 -u experiments/run_ablation.py --experiment A
    CUDA_VISIBLE_DEVICES=3 python3 -u experiments/run_ablation.py --experiment B
    CUDA_VISIBLE_DEVICES=3 python3 -u experiments/run_ablation.py --all

    # Score only (no GPU needed):
    python3 experiments/run_ablation.py --score-only --experiment A
    python3 experiments/run_ablation.py --score-only --all

    # Dry run:
    python3 experiments/run_ablation.py --dry-run --all
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from aq.config import load_run_config
from aq.eval.pipeline import read_records, run_inference

OUT_DIR = ROOT / "results"
TAU = 0.8
N_SEEDS = 5

MODEL_LABEL = "qwen25_3b"
MODEL_CONFIG = "configs/models/qwen25_3b.yaml"
MODEL_HF_ID = "Qwen/Qwen2.5-3B-Instruct"

DATASETS = [
    {"name": "mcqa_mmlu", "split": "test", "n_samples": 1500,
     "max_new_tokens": 48},
    {"name": "squad2", "split": "validation", "n_samples": 1500,
     "max_new_tokens": 128},
]

# ── condition definitions ──────────────────────────────────────────────

def _fp16():
    return {"label": "fp16", "config": "configs/quant/bf16.yaml",
            "overrides": {}}


def _ar(iters: int, variant: str = ""):
    suffix = f"_{variant}" if variant else "_pile"
    return {
        "label": f"ar{iters}{suffix}",
        "config": "configs/quant/autoround.yaml",
        "_variant": variant,
        "overrides": {
            "quant": {
                "extra": {
                    "hf_id": "__PLACEHOLDER__",
                    "iters": iters,
                    "nsamples": 128,
                    "seqlen": 2048,
                },
            },
        },
    }


EXP_A_CONDITIONS = [
    _fp16(),
    _ar(200),                       # pile-10k (cached from dose-response)
    _ar(200,  "calib_mcqa"),        # mcqa calibration (new)
    _ar(1000),                      # pile-10k (cached)
    _ar(1000, "calib_mcqa"),        # mcqa calibration (new)
]

EXP_B_CONDITIONS = [
    _fp16(),
    _ar(200),                       # both V+α+β (cached from dose-response)
    _ar(200, "rounding_only"),      # only V, α=1 β=1 (new)
    _ar(200, "clipping_only"),      # only α+β, V=0 (new)
]

EXP_A_DELTAS = [
    ("ar200_calib_mcqa",  "ar200_pile",  "mcqa vs pile at iters=200"),
    ("ar1000_calib_mcqa", "ar1000_pile", "mcqa vs pile at iters=1000"),
    ("ar200_pile",        "fp16",        "ar200 pile vs fp16"),
    ("ar200_calib_mcqa",  "fp16",        "ar200 mcqa vs fp16"),
    ("ar1000_pile",       "fp16",        "ar1000 pile vs fp16"),
    ("ar1000_calib_mcqa", "fp16",        "ar1000 mcqa vs fp16"),
]

EXP_B_DELTAS = [
    ("ar200_rounding_only",  "ar200_pile",          "rounding_only vs both"),
    ("ar200_clipping_only",  "ar200_pile",          "clipping_only vs both"),
    ("ar200_rounding_only",  "ar200_clipping_only", "rounding vs clipping"),
    ("ar200_pile",           "fp16",                "ar200 both vs fp16"),
    ("ar200_rounding_only",  "fp16",                "rounding_only vs fp16"),
    ("ar200_clipping_only",  "fp16",                "clipping_only vs fp16"),
]


# ── model path ─────────────────────────────────────────────────────────

def model_path(hf_id: str, iters: int, variant: str = "") -> str:
    name = hf_id.split("/")[-1]
    suffix = f"-{variant}" if variant else ""
    return f"models/{name}-AutoRound-iters{iters}{suffix}-Int4"


# ── config builder ─────────────────────────────────────────────────────

def build_cfg(qcfg: dict, dataset: dict, seed: int):
    overrides = {
        "max_new_tokens": dataset["max_new_tokens"],
        "batch_size": 32,
        **qcfg.get("overrides", {}),
    }
    if "__PLACEHOLDER__" in str(overrides):
        iters = overrides["quant"]["extra"]["iters"]
        variant = qcfg.get("_variant", "")
        overrides["quant"]["extra"]["hf_id"] = model_path(
            MODEL_HF_ID, iters, variant)
    return load_run_config(
        base="configs/base.yaml", model=MODEL_CONFIG,
        quant=qcfg["config"],
        data={"name": dataset["name"], "split": dataset["split"],
              "max_samples": dataset["n_samples"], "seed": seed},
        overrides=overrides,
    )


def release_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ── metrics (same as dose-response) ───────────────────────────────────

def auroc(scores: list[float], labels: list[int]) -> float | None:
    if not scores:
        return None
    arr = sorted(zip(scores, labels), key=lambda x: x[0])
    n_pos = sum(l for _, l in arr)
    n_neg = len(arr) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = [0.0] * len(arr)
    i = 0
    while i < len(arr):
        j = i
        while j < len(arr) and arr[j][0] == arr[i][0]:
            j += 1
        avg = (i + 1 + j) / 2
        for k in range(i, j):
            ranks[k] = avg
        i = j
    sum_pos = sum(r for r, (_, l) in zip(ranks, arr) if l == 1)
    return (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def ece(confs: list[float], accs: list[int], n_bins: int = 10) -> float | None:
    if not confs:
        return None
    c, a = np.array(confs), np.array(accs)
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        mask = ((c >= edges[i]) & (c < edges[i + 1]) if i < n_bins - 1
                else (c >= edges[i]) & (c <= edges[i + 1]))
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        total += (n_bin / len(c)) * abs(float(c[mask].mean()) -
                                        float(a[mask].mean()))
    return round(float(total), 4)


SIGNALS = [("verbalized", "verbalized_confidence"),
           ("token_prob", "token_prob_confidence")]


def full_metrics(records: list[dict], conf_key: str,
                 tau: float = TAU) -> dict:
    n_ans = n_correct = n_oc = n_parsed = 0
    confs, corrects = [], []
    for r in records:
        if r.get("abstained", False):
            continue
        n_ans += 1
        conf = (r.get("verbalized_confidence")
                if conf_key == "verbalized_confidence"
                else r.get("meta", {}).get("token_prob_confidence"))
        if r.get("correct"):
            n_correct += 1
        if conf is not None:
            n_parsed += 1
            confs.append(conf)
            corrects.append(1 if r.get("correct", False) else 0)
            if not r.get("correct", False) and conf > tau:
                n_oc += 1
    if n_ans == 0:
        return {"n_answered": 0}
    auc_val = auroc(confs, corrects)
    return {
        "n_answered": n_ans, "n_parsed": n_parsed,
        "accuracy": round(n_correct / n_ans, 5),
        "overconf": round(n_oc / n_ans, 5),
        "auroc": round(auc_val, 5) if auc_val is not None else None,
        "ece": ece(confs, corrects),
    }


def abstention_metrics(records: list[dict]) -> dict:
    n_unans = n_ans_q = n_abst = tp = ansq_correct = 0
    for r in records:
        answerable = bool(r.get("answerable", True))
        abstained = bool(r.get("abstained", False))
        correct = r.get("correct") is True
        if not answerable:
            n_unans += 1
        else:
            n_ans_q += 1
            if not abstained and correct:
                ansq_correct += 1
        if abstained:
            n_abst += 1
        if not answerable and abstained:
            tp += 1
    recall = tp / n_unans if n_unans else None
    precision = tp / n_abst if n_abst else None
    if recall is None or precision is None or (recall + precision) == 0:
        f1 = 0.0 if (recall is not None and precision is not None) else None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    acc_ansq = (ansq_correct / n_ans_q) if n_ans_q else None
    return {
        "abstention_f1": f1,
        "refuse_when_unanswerable": recall,
        "abstention_precision": precision,
        "accuracy_on_answerable": acc_ansq,
        "n_answerable": n_ans_q, "n_unanswerable": n_unans,
        "n_abstained": n_abst,
    }


def paired_t(deltas: list[float]) -> dict:
    if len(deltas) < 2:
        return {"mean": None, "note": "seeds < 2"}
    from aq.analysis.stats import mean_ci, one_sample_t
    t, p, df = one_sample_t(deltas, 0.0)
    m, lo, hi = mean_ci(deltas)
    return {
        "mean": round(float(np.mean(deltas)), 5),
        "t": round(t, 3), "p_two_sided": p, "df": df,
        "ci95_lo": round(lo, 5), "ci95_hi": round(hi, 5),
        "all_same_sign": bool(len(set(np.sign(deltas))) == 1),
        "per_seed": [round(v, 5) for v in deltas],
    }


def compute_deltas(seed_entries: list[dict],
                   key_a: str, key_b: str) -> dict:
    deltas: dict = {"seeds_used": [e["seed"] for e in seed_entries]}
    for sname, _ in SIGNALS:
        prefix = "verb" if sname == "verbalized" else "tp"
        d_acc, d_oc, d_auroc, d_ece = [], [], [], []
        for e in seed_entries:
            a, b = e[sname].get(key_a, {}), e[sname].get(key_b, {})
            if a.get("n_answered", 0) == 0 or b.get("n_answered", 0) == 0:
                continue
            d_acc.append(a["accuracy"] - b["accuracy"])
            d_oc.append(a["overconf"] - b["overconf"])
            if a.get("auroc") is not None and b.get("auroc") is not None:
                d_auroc.append(a["auroc"] - b["auroc"])
            if a.get("ece") is not None and b.get("ece") is not None:
                d_ece.append(a["ece"] - b["ece"])
        deltas[f"delta_{prefix}_accuracy"] = paired_t(d_acc)
        deltas[f"delta_{prefix}_overconf"] = paired_t(d_oc)
        deltas[f"delta_{prefix}_auroc"] = paired_t(d_auroc)
        deltas[f"delta_{prefix}_ece"] = paired_t(d_ece)

    d_f1, d_acca = [], []
    for e in seed_entries:
        aa = e["abstention"].get(key_a, {})
        bb = e["abstention"].get(key_b, {})
        if (aa.get("abstention_f1") is not None and
                bb.get("abstention_f1") is not None):
            d_f1.append(aa["abstention_f1"] - bb["abstention_f1"])
        if (aa.get("accuracy_on_answerable") is not None and
                bb.get("accuracy_on_answerable") is not None):
            d_acca.append(
                aa["accuracy_on_answerable"] - bb["accuracy_on_answerable"])
    deltas["delta_abstention_f1"] = paired_t(d_f1)
    deltas["delta_accuracy_on_answerable"] = paired_t(d_acca)
    return deltas


# ── generation ─────────────────────────────────────────────────────────

def generate(conditions: list[dict], dry_run: bool):
    total = len(conditions) * len(DATASETS) * N_SEEDS
    done = 0
    for qcfg in conditions:
        for dset in DATASETS:
            for seed in range(N_SEEDS):
                cfg = build_cfg(qcfg, dset, seed)
                p = cfg.raw_path()
                tag = f"{qcfg['label']}_{dset['name']}_s{seed}"
                done += 1
                if p.exists() and p.stat().st_size > 0:
                    n_rec = sum(1 for _ in open(p))
                    print(f"  [{done}/{total}] [CACHED] {tag}: "
                          f"{n_rec} records")
                    continue
                if dry_run:
                    print(f"  [{done}/{total}] [NEW]    {tag} -> "
                          f"{cfg.run_id()}")
                    continue
                print(f"\n>>> [{done}/{total}] {tag}")
                print(f"    run_id: {cfg.run_id()}")
                run_inference(cfg)
                release_gpu()


# ── scoring ────────────────────────────────────────────────────────────

def score(conditions: list[dict], delta_pairs: list[tuple],
          experiment_name: str, design_desc: dict) -> dict:
    labels = [c["label"] for c in conditions]

    result: dict = {
        "experiment": experiment_name,
        "design": design_desc,
        "n_seeds": N_SEEDS, "tau": TAU,
        "datasets": {},
    }

    for dset in DATASETS:
        dname = dset["name"]
        seed_data: list = []

        for seed in range(N_SEEDS):
            paths = {}
            run_ids = {}
            for qcfg in conditions:
                cfg = build_cfg(qcfg, dset, seed)
                paths[qcfg["label"]] = cfg.raw_path()
                run_ids[qcfg["label"]] = cfg.run_id()

            missing = [k for k, p in paths.items() if not p.exists()]
            if missing:
                print(f"  [MISSING] {dname}/seed{seed}: {missing}")
                continue

            recs = {k: read_records(p) for k, p in paths.items()}
            entry: dict = {"seed": seed, "run_ids": run_ids}

            for sname, skey in SIGNALS:
                entry[sname] = {}
                for label in labels:
                    entry[sname][label] = full_metrics(recs[label], skey)
            entry["abstention"] = {}
            for label in labels:
                entry["abstention"][label] = abstention_metrics(recs[label])

            seed_data.append(entry)

        ds_result: dict = {"per_seed": seed_data}
        if seed_data:
            ds_result["deltas"] = {}
            for key_a, key_b, desc in delta_pairs:
                has_a = any(key_a in e.get("verbalized", {})
                            for e in seed_data)
                has_b = any(key_b in e.get("verbalized", {})
                            for e in seed_data)
                if has_a and has_b:
                    ds_result["deltas"][f"{key_a}_vs_{key_b}"] = \
                        compute_deltas(seed_data, key_a, key_b)
                else:
                    print(f"  [SKIP delta] {desc}: missing data")
        result["datasets"][dname] = ds_result

    return result


def print_summary(result: dict):
    print(f"\n{'=' * 65}")
    print(f"  {result['experiment']}")
    print(f"  model: {result['design'].get('model', '?')}")
    print(f"{'=' * 65}")

    for dname, ds in result.get("datasets", {}).items():
        print(f"\n  --- {dname} ---")
        for comp_name, comp in ds.get("deltas", {}).items():
            m = comp.get("delta_verb_auroc", {}).get("mean")
            p = comp.get("delta_verb_auroc", {}).get("p_two_sided")
            oc = comp.get("delta_verb_overconf", {}).get("mean")
            if m is not None:
                sig = "*" if (p or 1) < 0.05 else " "
                print(f"    {comp_name:40s}: Δverb_auroc={m:+.4f}{sig}"
                      f"  Δoverconf={oc:+.4f}")


# ── main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ablation: calibration data (A) & rounding/clipping (B)")
    parser.add_argument("--experiment", choices=["A", "B", "all"],
                        default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()

    experiments = []

    if args.experiment in ("A", "all"):
        experiments.append({
            "name": "calibration_data_ablation",
            "conditions": EXP_A_CONDITIONS,
            "deltas": EXP_A_DELTAS,
            "design": {
                "variable_isolated": "Calibration data content "
                                     "(pile-10k vs MCQA prompts)",
                "levels": {
                    "ar200_pile": "iters=200, calib=pile-10k",
                    "ar200_calib_mcqa": "iters=200, calib=MCQA",
                    "ar1000_pile": "iters=1000, calib=pile-10k",
                    "ar1000_calib_mcqa": "iters=1000, calib=MCQA",
                },
                "controlled": [
                    "library (AutoRound)", "bits (4)",
                    "group_size (128)", "nsamples (128)", "seqlen (2048)",
                    "format (auto_gptq)", "loader (GPTQModel)",
                ],
                "model": MODEL_LABEL,
                "baseline": "fp16",
            },
            "out_file": f"ablation_calib_data_{MODEL_LABEL}.json",
        })

    if args.experiment in ("B", "all"):
        experiments.append({
            "name": "rounding_vs_clipping_decomposition",
            "conditions": EXP_B_CONDITIONS,
            "deltas": EXP_B_DELTAS,
            "design": {
                "variable_isolated": "Optimization target "
                                     "(V rounding vs α,β clipping)",
                "levels": {
                    "ar200_pile": "iters=200, both V+α+β (default)",
                    "ar200_rounding_only": "iters=200, V only (α=1,β=1)",
                    "ar200_clipping_only": "iters=200, α+β only (V=0)",
                },
                "controlled": [
                    "library (AutoRound)", "bits (4)",
                    "group_size (128)", "iters (200)",
                    "calibration (pile-10k)", "nsamples (128)",
                    "seqlen (2048)", "format (auto_gptq)",
                ],
                "model": MODEL_LABEL,
                "baseline": "fp16",
            },
            "out_file": f"ablation_decompose_{MODEL_LABEL}.json",
        })

    for exp in experiments:
        print(f"\n{'#' * 65}")
        print(f"# {exp['name']}")
        print(f"# Conditions: "
              f"{[c['label'] for c in exp['conditions']]}")
        print(f"{'#' * 65}")

        if not args.score_only:
            print("\nPhase 1: Generation")
            print("-" * 40)
            generate(exp["conditions"], args.dry_run)

        if args.dry_run:
            continue

        print("\nPhase 2: Scoring")
        print("-" * 40)
        result = score(exp["conditions"], exp["deltas"],
                       exp["name"], exp["design"])

        out_path = OUT_DIR / exp["out_file"]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n>>> Saved: {out_path}")
        print_summary(result)


if __name__ == "__main__":
    main()
