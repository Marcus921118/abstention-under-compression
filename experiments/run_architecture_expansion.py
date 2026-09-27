#!/usr/bin/env python3
"""#24 — 架構軸擴充:Llama-3.2-1B/3B + Gemma-3-4B。

動機:既有結果只在 Qwen2.5 + Phi-3 上成立。新增三個模型(大小對齊既有 roster)
以測試結論是否跨架構。

設計:
  - Llama-3.2-1B ↔ Qwen2.5-1.5B（小型）
  - Llama-3.2-3B ↔ Qwen2.5-3B（中型）
  - Gemma-3-4B   ↔ Phi-3-mini-3.8B（中型）
  - 差異只來自架構,不混進規模

實驗矩陣:
  3 models × 2 quants (fp16, bnb_nf4) × 2 datasets (mcqa_mmlu, squad2) × 5 seeds
  = 60 runs

計算指標(逐 seed 配對 Δ,全部對 fp16 基準):
  信心通道相關(verbalized + token_prob):
    Δ Accuracy, Δ OverconfVerb, Verb Δ AUROC,
    Δ TP Overconf, TP Δ ECE, TP Δ AUROC
  棄答相關(訊號無關):
    Δ Abs_F1, Δ Precision, Δ Recall

前置:vertical_slice_newarch.py 必須全部 PASS。

用法:
    python3 experiments/run_architecture_expansion.py --dry-run
    CUDA_VISIBLE_DEVICES=0 python3 experiments/run_architecture_expansion.py
    CUDA_VISIBLE_DEVICES=0 python3 experiments/run_architecture_expansion.py \\
        --model llama32_1b --dataset mcqa_mmlu
    python3 experiments/run_architecture_expansion.py --score-only
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np

from aq.config import load_run_config
from aq.eval.pipeline import read_records, run_inference

OUT_DIR = ROOT / "results"
TAU = 0.8
N_SEEDS = 5

MODELS = [
    {"label": "llama32_1b", "config": "configs/models/llama32_1b.yaml",
     "paired_with": "qwen25_15b", "size_class": "small"},
    {"label": "llama32_3b", "config": "configs/models/llama32_3b.yaml",
     "paired_with": "qwen25_3b", "size_class": "medium"},
    {"label": "gemma3_4b",  "config": "configs/models/gemma3_4b.yaml",
     "paired_with": "phi3_mini", "size_class": "medium"},
]

DATASETS = [
    {"name": "mcqa_mmlu", "split": "test", "n_samples": 1500,
     "max_new_tokens": 48},
    {"name": "squad2", "split": "validation", "n_samples": 1500,
     "max_new_tokens": 128},
]

QUANTS = [
    {"label": "bf16",    "config": "configs/quant/bf16.yaml"},  # <--- 改為 bf16
    {"label": "bnb_nf4", "config": "configs/quant/bnb_nf4.yaml"},
]

SIGNALS = [("verbalized", "verbalized_confidence"),
           ("token_prob", "token_prob_confidence")]


# ── config helper ─────────────────────────────────────────────────────

def build_cfg(model_yaml: str, quant_yaml: str, dataset: dict, seed: int):
    return load_run_config(
        base="configs/base.yaml", model=model_yaml, quant=quant_yaml,
        data={"name": dataset["name"], "split": dataset["split"],
              "max_samples": dataset["n_samples"], "seed": seed},
        overrides={
            "max_new_tokens": dataset["max_new_tokens"],
            "batch_size": 32  # <--- 加在這裡！把卡車裝滿！
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


# ── metrics ───────────────────────────────────────────────────────────

def full_metrics(records: list[dict], conf_key: str, tau: float = TAU) -> dict:
    """信心通道指標:overconf + AUROC + ECE。沿用 run_squad2_headline 的定義。"""
    n_ans = n_correct = n_oc = n_parsed = 0
    confs, corrects, answerables = [], [], []
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
            answerables.append(1 if r.get("answerable", True) else 0)
            if not r.get("correct", False) and conf > tau:
                n_oc += 1
    if n_ans == 0:
        return {"n_answered": 0}
    from run_squad2_headline import auroc, ece
    auc_val = auroc(confs, corrects)
    return {
        "n_answered": n_ans, "n_parsed": n_parsed,
        "accuracy": round(n_correct / n_ans, 5),
        "overconf": round(n_oc / n_ans, 5),
        "auroc": round(auc_val, 5) if auc_val is not None else None,
        "ece": ece(confs, corrects),
    }


def abstention_metrics_from_records(records: list[dict]) -> dict:
    """棄答四指標(訊號無關),直接從 raw records 算。
    與 rescore_abstention.py 的 abstention_metrics() 邏輯完全一致。"""
    n_unans = 0
    n_ans_q = 0
    n_abst = 0
    tp = 0
    ansq_correct = 0

    for r in records:
        answerable = bool(r.get("answerable", True))
        abstained = bool(r.get("abstained", False))
        correct = r.get("correct") is True

        # 計算不可答題 (Unanswerable) 與 可答題 (Answerable)
        if not answerable:
            n_unans += 1
        else:
            n_ans_q += 1
            if not abstained and correct:
                ansq_correct += 1
                
        # 計算總棄答數
        if abstained:
            n_abst += 1
            
        # 計算 True Positive (在不可答題上棄答)
        if not answerable and abstained:
            tp += 1

    # 計算 A2, A3 (分母為 0 時回傳 None，觸發該 seed 排除)
    recall = tp / n_unans if n_unans else None
    precision = tp / n_abst if n_abst else None
    
    # 計算 A1 (F1 Score)
    if recall is None or precision is None or (recall + precision) == 0:
        f1 = 0.0 if (recall is not None and precision is not None) else None
    else:
        f1 = 2 * precision * recall / (precision + recall)
        
    # 計算 A4 (Accuracy on Answerable)
    acc_ansq = (ansq_correct / n_ans_q) if n_ans_q else None

    return {
        "abstention_f1": f1,
        "refuse_when_unanswerable": recall,
        "abstention_precision": precision,
        "accuracy_on_answerable": acc_ansq,
        "n_answerable": n_ans_q, 
        "n_unanswerable": n_unans,
        "n_abstained": n_abst, 
        "tp_abstain_on_unanswerable": tp,
    }


# ── paired t-test ─────────────────────────────────────────────────────

def paired_t(deltas: list[float]) -> dict:
    """逐 seed Δ 的配對 t 檢定。"""
    if len(deltas) < 2:
        return {"mean": None, "note": "seeds 不足"}
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


# ── generation phase ──────────────────────────────────────────────────

def generate(models_to_run, datasets_to_run, n_seeds, dry_run):
    total = len(models_to_run) * len(QUANTS) * len(datasets_to_run) * n_seeds
    done = 0
    for mcfg in models_to_run:
        for dset in datasets_to_run:
            for qcfg in QUANTS:
                for seed in range(n_seeds):
                    cfg = build_cfg(mcfg["config"], qcfg["config"], dset, seed)
                    p = cfg.raw_path()
                    tag = f"{mcfg['label']}_{qcfg['label']}_{dset['name']}_s{seed}"
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


# ── scoring phase ─────────────────────────────────────────────────────

def score(models_to_run, datasets_to_run, n_seeds) -> dict:
    """計算全部 9 個 Δ 指標。"""
    results: dict = {"models": {}}

    for mcfg in models_to_run:
        model_results: dict = {"datasets": {}, "meta": {
            "paired_with": mcfg["paired_with"],
            "size_class": mcfg["size_class"],
        }}

        for dset in datasets_to_run:
            dname = dset["name"]
            seed_data: dict = {"per_seed": [], "deltas": {}}

            for seed in range(n_seeds):
                bf16_cfg = build_cfg(mcfg["config"], QUANTS[0]["config"], dset, seed)
                nf4_cfg = build_cfg(mcfg["config"], QUANTS[1]["config"], dset, seed)
                bf16_p, nf4_p = bf16_cfg.raw_path(), nf4_cfg.raw_path()

                if not bf16_p.exists() or not nf4_p.exists():
                    print(f"  [MISSING] {mcfg['label']}/{dname}/seed{seed}")
                    continue

                bf16_recs = read_records(bf16_p)
                nf4_recs = read_records(nf4_p)

                entry = {
                    "seed": seed,
                    "bf16_run_id": bf16_cfg.run_id(),
                    "nf4_run_id": nf4_cfg.run_id(),
                }

                # 信心通道指標
                for sname, skey in SIGNALS:
                    bf16_m = full_metrics(bf16_recs, skey, tau=TAU)
                    nf4_m = full_metrics(nf4_recs, skey, tau=TAU)
                    entry[sname] = {"bf16": bf16_m, "nf4": nf4_m}

                # 棄答指標(訊號無關)
                bf16_abst = abstention_metrics_from_records(bf16_recs)
                nf4_abst = abstention_metrics_from_records(nf4_recs)
                entry["abstention"] = {"bf16": bf16_abst, "nf4": nf4_abst}

                seed_data["per_seed"].append(entry)

            # ── 跨 seed 聚合(配對 Δ)──────────────────────
            seeds_used = [e["seed"] for e in seed_data["per_seed"]]
            deltas: dict = {"seeds_used": seeds_used}

            # 信心通道 Δ
            for sname, _ in SIGNALS:
                prefix = "verb" if sname == "verbalized" else "tp"
                d_acc, d_oc, d_auroc, d_ece = [], [], [], []
                for e in seed_data["per_seed"]:
                    bf, nf = e[sname]["bf16"], e[sname]["nf4"]
                    if bf.get("n_answered", 0) == 0:
                        continue
                    d_acc.append(nf["accuracy"] - bf["accuracy"])
                    d_oc.append(nf["overconf"] - bf["overconf"])
                    if nf.get("auroc") is not None and bf.get("auroc") is not None:
                        d_auroc.append(nf["auroc"] - bf["auroc"])
                    if nf.get("ece") is not None and bf.get("ece") is not None:
                        d_ece.append(nf["ece"] - bf["ece"])

                # ... (這部分計算 delta 的程式碼不動) ...

            # 棄答 Δ(訊號無關)
            d_f1, d_prec, d_rec = [], [], []
            for e in seed_data["per_seed"]:
                bf_a, nf_a = e["abstention"]["bf16"], e["abstention"]["nf4"]
                if bf_a.get("abstention_f1") is not None and nf_a.get("abstention_f1") is not None:
                    d_f1.append(nf_a["abstention_f1"] - bf_a["abstention_f1"])
                if (bf_a.get("abstention_precision") is not None
                        and nf_a.get("abstention_precision") is not None):
                    d_prec.append(nf_a["abstention_precision"] - bf_a["abstention_precision"])
                if (bf_a.get("refuse_when_unanswerable") is not None
                        and nf_a.get("refuse_when_unanswerable") is not None):
                    d_rec.append(nf_a["refuse_when_unanswerable"] - bf_a["refuse_when_unanswerable"])

            deltas["delta_abstention_f1"] = paired_t(d_f1)
            deltas["delta_abstention_precision"] = paired_t(d_prec)
            deltas["delta_abstention_recall"] = paired_t(d_rec)

            seed_data["deltas"] = deltas
            model_results["datasets"][dname] = seed_data

            # ── 印摘要 ──
            print(f"\n  {mcfg['label']} / {dname}:")
            for key, label_str in [
                ("delta_accuracy_verb", "Δ Accuracy(verb)"),
                ("delta_overconf_verb", "Δ OverconfVerb"),
                ("delta_auroc_verb", "Verb Δ AUROC"),
                ("delta_overconf_tp", "Δ TP Overconf"),
                ("delta_ece_tp", "TP Δ ECE"),
                ("delta_auroc_tp", "TP Δ AUROC"),
                ("delta_abstention_f1", "Δ Abs_F1"),
                ("delta_abstention_precision", "Δ Precision"),
                ("delta_abstention_recall", "Δ Recall"),
            ]:
                d = deltas.get(key, {})
                m = d.get("mean")
                if m is not None:
                    p = d.get("p_two_sided", "?")
                    print(f"    {label_str:<20} {m:+.5f}  p={p:.4g}" if isinstance(p, float)
                          else f"    {label_str:<20} {m:+.5f}  p={p}")
                else:
                    print(f"    {label_str:<20} N/A")

        results["models"][mcfg["label"]] = model_results
    return results


# ── main ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=[m["label"] for m in MODELS], default=None)
    ap.add_argument("--dataset", choices=[d["name"] for d in DATASETS], default=None)
    ap.add_argument("--n-seeds", type=int, default=N_SEEDS)
    ap.add_argument("--n-samples", type=int, default=None,
                    help="覆寫 n_samples(冒煙測用)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--out-name", default="architecture_expansion_p1")
    args = ap.parse_args()

    models_to_run = ([m for m in MODELS if m["label"] == args.model]
                     if args.model else MODELS)
    datasets_to_run = ([d for d in DATASETS if d["name"] == args.dataset]
                       if args.dataset else DATASETS)

    if args.n_samples is not None:
        for d in datasets_to_run:
            d["n_samples"] = args.n_samples

    n_seeds = args.n_seeds
    total = len(models_to_run) * len(QUANTS) * len(datasets_to_run) * n_seeds

    # 冒煙測保護
    is_smoke = (args.n_samples is not None and args.n_samples < 1500) or n_seeds < 5
    if is_smoke and "smoke" not in args.out_name:
        print("⚠️  非正式規模。--out-name 必須含 'smoke'。")
        ap.error("縮減規模時 --out-name 必須含 'smoke'")

    print("=" * 66)
    print("#24 ARCHITECTURE EXPANSION")
    print(f"  models:   {[m['label'] for m in models_to_run]}")
    print(f"  datasets: {[d['name'] for d in datasets_to_run]}")
    print(f"  seeds:    {n_seeds}")
    print(f"  n_samples: {[d['n_samples'] for d in datasets_to_run]}")
    print(f"  total runs: {total}")
    est_min = total * 25
    print(f"  est. time:  ~{est_min} min (~{est_min/60:.1f} hrs)")
    print("=" * 66)

    # vertical slice gate
    vs_path = ROOT / "results" / "vertical_slice_newarch.json"
    if vs_path.exists():
        vs = json.loads(vs_path.read_text(encoding="utf-8"))
        for mcfg in models_to_run:
            mdata = vs.get("models", {}).get(mcfg["label"], {})
            gates = mdata.get("gates", {})
            failed = [k for k, v in gates.items() if not v.get("pass", True)]
            if failed:
                print(f"🔴 {mcfg['label']} vertical slice 有未通過的 gate: {failed}")
                print(f"   先修復 vertical_slice_newarch.py 的問題。")
                sys.exit(1)
        print("✓ vertical slice 全部 PASS\n")
    else:
        print("⚠️  vertical_slice_newarch.json 不存在。")
        print("   建議先跑 python3 experiments/vertical_slice_newarch.py\n")

    if not args.score_only:
        generate(models_to_run, datasets_to_run, n_seeds, args.dry_run)
        if args.dry_run:
            return

    print(f"\n{'=' * 66}")
    print("SCORING")
    print(f"{'=' * 66}")
    results = score(models_to_run, datasets_to_run, n_seeds)

    # ── check completeness ──
    incomplete = []
    for mlabel, mdata in results["models"].items():
        for dname, ddata in mdata["datasets"].items():
            n = len(ddata.get("deltas", {}).get("seeds_used", []))
            if n < n_seeds:
                incomplete.append(f"{mlabel}/{dname}: {n}/{n_seeds} seeds")

    out_name = args.out_name
    if incomplete:
        out_name = f"{args.out_name}_INCOMPLETE"
        print(f"\n🔴 SEEDS 不足(鐵則 3):")
        for msg in incomplete:
            print(f"   {msg}")

    out = {
        "experiment": "architecture_expansion",
        "design": {
            "size_matching": {
                "llama32_1b": "qwen25_15b (small)",
                "llama32_3b": "qwen25_3b (medium)",
                "gemma3_4b": "phi3_mini (medium)",
            },
            "purpose": "isolate architecture effect, control for scale",
        },
        "n_seeds": n_seeds,
        "tau": TAU,
        "datasets": {d["name"]: {"n_samples": d["n_samples"],
                                  "max_new_tokens": d["max_new_tokens"]}
                     for d in datasets_to_run},
        **results,
    }
    out_path = OUT_DIR / f"{out_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n✓ Results: {out_path}")
    if not incomplete:
        print("  可進 FDR 家族。")


if __name__ == "__main__":
    main()
