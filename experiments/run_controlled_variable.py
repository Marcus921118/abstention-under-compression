#!/usr/bin/env python3
"""#28 Controlled-variable experiments: isolating single quantization design variables.

Two experiments, each differing in exactly ONE design variable:

  Exp A: "codebook" — bnb NF4 vs bnb FP4
    Shared: library (bnb), blocksize (64), scale (absmax), rounding (RTN),
            double_quant (true), compute_dtype (bfloat16), bits (4).
    Variable: codebook — NF4 (N(0,1) quantiles) vs FP4 (E2M1 floating-point).

  Exp B: "rounding" — GPTQ vs RTN
    Shared: bits (4), group_size (128), uniform integer grid.
    Variable: rounding strategy — GPTQ (Hessian OBQ + calibration) vs
              RTN (simple round-to-nearest, no calibration).

Design per experiment:
  model × {fp16, method_a, method_b} × {mcqa_mmlu, squad2} × 5 seeds × n=1500
  Cache-first: raw JSONL 存在就跳過生成。

Pre-requisites (Exp B only):
    # RTN pre-quantized models — run BEFORE the main experiment:
    CUDA_VISIBLE_DEVICES=0 python3 -u experiments/prequantize_rtn.py \\
        --model Qwen/Qwen2.5-3B-Instruct
    CUDA_VISIBLE_DEVICES=0 python3 -u experiments/prequantize_rtn.py \\
        --model Qwen/Qwen2.5-1.5B-Instruct

Usage:
    # Dry run (印快取狀態,不跑模型):
    python3 experiments/run_controlled_variable.py --experiment codebook --dry-run
    python3 experiments/run_controlled_variable.py --experiment rounding --dry-run

    # GPU server 跑實驗:
    CUDA_VISIBLE_DEVICES=0 python3 -u experiments/run_controlled_variable.py \\
        --experiment codebook
    CUDA_VISIBLE_DEVICES=0 python3 -u experiments/run_controlled_variable.py \\
        --experiment rounding

    # 只算指標(所有 raw 已跑完):
    python3 experiments/run_controlled_variable.py --experiment codebook --score-only
    python3 experiments/run_controlled_variable.py --experiment all --score-only

    # 只跑特定 model/dataset:
    python3 -u experiments/run_controlled_variable.py \\
        --experiment codebook --model qwen25_15b --dataset mcqa_mmlu
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

# ── model configs ────────────────────────────────────────────────────

MODEL_CONFIGS = {
    "qwen25_3b": {
        "config": "configs/models/qwen25_3b.yaml",
        "gptq_hf_id": "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4",
        "rtn_hf_id": "models/Qwen2.5-3B-Instruct-RTN-Int4",
    },
    "qwen25_15b": {
        "config": "configs/models/qwen25_15b.yaml",
        "gptq_hf_id": "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4",
        "rtn_hf_id": "models/Qwen2.5-1.5B-Instruct-RTN-Int4",
    },
}

DATASETS = [
    {"name": "mcqa_mmlu", "split": "test", "n_samples": 1500,
     "max_new_tokens": 48},
    {"name": "squad2", "split": "validation", "n_samples": 1500,
     "max_new_tokens": 128},
]

# ── experiment definitions ───────────────────────────────────────────

def build_codebook_quants() -> list[dict]:
    return [
        {"label": "fp16",    "config": "configs/quant/bf16.yaml",
         "overrides": {}},
        {"label": "bnb_nf4", "config": "configs/quant/bnb_nf4.yaml",
         "overrides": {}},
        {"label": "bnb_fp4", "config": "configs/quant/bnb_fp4.yaml",
         "overrides": {}},
    ]


def build_rounding_quants(model_label: str) -> list[dict]:
    mc = MODEL_CONFIGS[model_label]
    return [
        {"label": "fp16",  "config": "configs/quant/bf16.yaml",
         "overrides": {}},
        {"label": "gptq",  "config": "configs/quant/gptq.yaml",
         "overrides": {"quant": {"extra": {"hf_id": mc["gptq_hf_id"]}}}},
        {"label": "rtn",   "config": "configs/quant/rtn.yaml",
         "overrides": {"quant": {"extra": {"hf_id": mc["rtn_hf_id"]}}}},
    ]


EXPERIMENTS = {
    "codebook": {
        "description": "NF4 vs FP4 — codebook design (1 variable)",
        "variable": "codebook design",
        "method_a": "bnb_nf4",
        "method_b": "bnb_fp4",
        "method_a_desc": "NF4 (N(0,1) quantiles, non-uniform)",
        "method_b_desc": "FP4 (E2M1 floating-point, non-uniform)",
        "controlled": [
            "library (bitsandbytes)", "blocksize (64, bnb default)",
            "scale (absmax per block)", "rounding (RTN)",
            "double_quant (true)", "compute_dtype (bfloat16)", "bits (4)",
        ],
        "build_quants": lambda model_label: build_codebook_quants(),
    },
    "rounding": {
        "description": "GPTQ vs RTN — rounding strategy (1 variable)",
        "variable": "rounding strategy",
        "method_a": "gptq",
        "method_b": "rtn",
        "method_a_desc": "GPTQ (Hessian OBQ, cross-weight compensation, calibration data)",
        "method_b_desc": "RTN (round-to-nearest, no compensation, no calibration)",
        "controlled": [
            "bits (4)", "group_size (128)", "grid (uniform integer)",
            "output format (GPTQ-compatible)",
        ],
        "build_quants": lambda model_label: build_rounding_quants(model_label),
    },
}


# ── metrics helpers (self-contained) ─────────────────────────────────

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
        total += (n_bin / len(c)) * abs(float(c[mask].mean()) - float(a[mask].mean()))
    return round(float(total), 4)


def full_metrics(records: list[dict], conf_key: str, tau: float = TAU) -> dict:
    n_ans = n_correct = n_oc = n_parsed = 0
    confs, corrects = [], []
    for r in records:
        if r.get("abstained", False):
            continue
        n_ans += 1
        conf = (r.get("verbalized_confidence") if conf_key == "verbalized_confidence"
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
        "n_answerable": n_ans_q,
        "n_unanswerable": n_unans,
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


# ── config builder ───────────────────────────────────────────────────

def build_cfg(model_yaml: str, qcfg: dict, dataset: dict, seed: int):
    return load_run_config(
        base="configs/base.yaml", model=model_yaml, quant=qcfg["config"],
        data={"name": dataset["name"], "split": dataset["split"],
              "max_samples": dataset["n_samples"], "seed": seed},
        overrides={
            "max_new_tokens": dataset["max_new_tokens"],
            "batch_size": 32,
            **qcfg.get("overrides", {}),
        },
    )


def release_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ── phase 1: generation ─────────────────────────────────────────────

def generate(model_yaml: str, quants: list[dict], datasets: list[dict],
             n_seeds: int, dry_run: bool):
    total = len(quants) * len(datasets) * n_seeds
    done = 0
    for qcfg in quants:
        for dset in datasets:
            for seed in range(n_seeds):
                cfg = build_cfg(model_yaml, qcfg, dset, seed)
                p = cfg.raw_path()
                tag = f"{qcfg['label']}_{dset['name']}_s{seed}"
                done += 1
                if p.exists() and p.stat().st_size > 0:
                    n_rec = sum(1 for _ in open(p))
                    print(f"  [{done}/{total}] [CACHED] {tag}: {n_rec} records")
                    continue
                if dry_run:
                    print(f"  [{done}/{total}] [NEW]    {tag} -> {cfg.run_id()}")
                    continue
                print(f"\n>>> [{done}/{total}] {tag}")
                print(f"    run_id: {cfg.run_id()}")
                run_inference(cfg)
                release_gpu()


# ── phase 2: scoring ────────────────────────────────────────────────

SIGNALS = [("verbalized", "verbalized_confidence"),
           ("token_prob", "token_prob_confidence")]


def compute_deltas(seed_entries: list[dict], key_a: str, key_b: str) -> dict:
    """Compute paired deltas (method_a - method_b) across seeds."""
    deltas: dict = {"seeds_used": [e["seed"] for e in seed_entries]}
    for sname, _ in SIGNALS:
        prefix = "verb" if sname == "verbalized" else "tp"
        d_acc, d_oc, d_auroc, d_ece = [], [], [], []
        for e in seed_entries:
            a, b = e[sname][key_a], e[sname][key_b]
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

    d_f1, d_prec, d_rec, d_acca = [], [], [], []
    for e in seed_entries:
        aa, bb = e["abstention"][key_a], e["abstention"][key_b]
        if aa.get("abstention_f1") is not None and bb.get("abstention_f1") is not None:
            d_f1.append(aa["abstention_f1"] - bb["abstention_f1"])
        if aa.get("abstention_precision") is not None and bb.get("abstention_precision") is not None:
            d_prec.append(aa["abstention_precision"] - bb["abstention_precision"])
        if aa.get("refuse_when_unanswerable") is not None and bb.get("refuse_when_unanswerable") is not None:
            d_rec.append(aa["refuse_when_unanswerable"] - bb["refuse_when_unanswerable"])
        if aa.get("accuracy_on_answerable") is not None and bb.get("accuracy_on_answerable") is not None:
            d_acca.append(aa["accuracy_on_answerable"] - bb["accuracy_on_answerable"])
    deltas["delta_abstention_f1"] = paired_t(d_f1)
    deltas["delta_abstention_precision"] = paired_t(d_prec)
    deltas["delta_abstention_recall"] = paired_t(d_rec)
    deltas["delta_accuracy_on_answerable"] = paired_t(d_acca)
    return deltas


def score(exp_name: str, model_label: str, model_yaml: str,
          quants: list[dict], datasets: list[dict], n_seeds: int) -> dict:
    exp = EXPERIMENTS[exp_name]
    fp16_qcfg = quants[0]
    assert fp16_qcfg["label"] == "fp16"
    qcfg_a = next(q for q in quants if q["label"] == exp["method_a"])
    qcfg_b = next(q for q in quants if q["label"] == exp["method_b"])
    label_a, label_b = exp["method_a"], exp["method_b"]

    result: dict = {
        "experiment": f"controlled_{exp_name}",
        "design": {
            "variable_isolated": exp["variable"],
            "method_a": f"{label_a} ({exp['method_a_desc']})",
            "method_b": f"{label_b} ({exp['method_b_desc']})",
            "controlled": exp["controlled"],
            "model": model_label,
            "baseline": "fp16 (bf16)",
        },
        "n_seeds": n_seeds,
        "tau": TAU,
        "datasets": {},
    }

    for dset in datasets:
        dname = dset["name"]
        seed_data: list = []

        for seed in range(n_seeds):
            cfg_fp16 = build_cfg(model_yaml, fp16_qcfg, dset, seed)
            cfg_a = build_cfg(model_yaml, qcfg_a, dset, seed)
            cfg_b = build_cfg(model_yaml, qcfg_b, dset, seed)

            paths = {
                "fp16": cfg_fp16.raw_path(),
                label_a: cfg_a.raw_path(),
                label_b: cfg_b.raw_path(),
            }
            missing = [k for k, p in paths.items() if not p.exists()]
            if missing:
                print(f"  [MISSING] {dname}/seed{seed}: {missing}")
                continue

            recs = {k: read_records(p) for k, p in paths.items()}
            entry: dict = {"seed": seed}
            for k in ("fp16", label_a, label_b):
                entry.setdefault("run_ids", {})[k] = (
                    build_cfg(model_yaml,
                              {"fp16": fp16_qcfg, label_a: qcfg_a,
                               label_b: qcfg_b}[k],
                              dset, seed).run_id()
                )

            for sname, skey in SIGNALS:
                entry[sname] = {}
                for k in ("fp16", label_a, label_b):
                    entry[sname][k] = full_metrics(recs[k], skey, tau=TAU)
            entry["abstention"] = {}
            for k in ("fp16", label_a, label_b):
                entry["abstention"][k] = abstention_metrics(recs[k])

            seed_data.append(entry)

        ds_result: dict = {"per_seed": seed_data}
        if seed_data:
            ds_result["deltas_a_vs_b"] = compute_deltas(
                seed_data, label_a, label_b)
            ds_result["deltas_a_vs_fp16"] = compute_deltas(
                seed_data, label_a, "fp16")
            ds_result["deltas_b_vs_fp16"] = compute_deltas(
                seed_data, label_b, "fp16")
        result["datasets"][dname] = ds_result

    return result


# ── summary ──────────────────────────────────────────────────────────

def print_summary(result: dict):
    exp = result["experiment"]
    design = result["design"]
    print(f"\n{'=' * 65}")
    print(f"  {exp}: {design['variable_isolated']}")
    print(f"  {design['method_a']}  vs  {design['method_b']}")
    print(f"  model: {design['model']}")
    print(f"{'=' * 65}")

    for dname, ds in result.get("datasets", {}).items():
        d_ab = ds.get("deltas_a_vs_b", {})
        print(f"\n  --- {dname} (A vs B direct) ---")
        for key in ("delta_verb_overconf", "delta_verb_ece",
                     "delta_verb_accuracy", "delta_verb_auroc",
                     "delta_abstention_f1", "delta_accuracy_on_answerable"):
            d = d_ab.get(key, {})
            m = d.get("mean")
            p = d.get("p_two_sided")
            if isinstance(m, (int, float)) and isinstance(p, (int, float)):
                sig = "*" if p < 0.05 else " "
                print(f"    {key:35s}: Δ={m:+.4f}  p={p:.4f} {sig}")
            else:
                note = d.get("note", "no data")
                print(f"    {key:35s}: ({note})")


# ── main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="#28 Controlled-variable experiments")
    parser.add_argument("--experiment", required=True,
                        choices=["codebook", "rounding", "all"],
                        help="which experiment to run")
    parser.add_argument("--model", default="qwen25_3b",
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--dry-run", action="store_true",
                        help="印 run_id 和快取狀態,不跑模型")
    parser.add_argument("--score-only", action="store_true",
                        help="跳過生成,只從 raw 算指標")
    parser.add_argument("--dataset", default=None,
                        choices=["mcqa_mmlu", "squad2"])
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    args = parser.parse_args()

    exps = (["codebook", "rounding"] if args.experiment == "all"
            else [args.experiment])
    datasets = ([d for d in DATASETS if d["name"] == args.dataset]
                if args.dataset else DATASETS)
    mc = MODEL_CONFIGS[args.model]
    model_yaml = mc["config"]

    for exp_name in exps:
        exp = EXPERIMENTS[exp_name]
        quants = exp["build_quants"](args.model)

        print(f"\n{'#' * 65}")
        print(f"# Experiment: {exp['description']}")
        print(f"# Model: {args.model}  Seeds: {args.seeds}")
        print(f"{'#' * 65}")

        if not args.score_only:
            print("\nPhase 1: Generation")
            print("-" * 40)
            generate(model_yaml, quants, datasets, args.seeds, args.dry_run)

        if args.dry_run:
            continue

        print("\nPhase 2: Scoring")
        print("-" * 40)
        result = score(exp_name, args.model, model_yaml,
                       quants, datasets, args.seeds)

        out_path = OUT_DIR / f"controlled_{exp_name}_{args.model}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n>>> Saved: {out_path}")

        print_summary(result)


if __name__ == "__main__":
    main()
