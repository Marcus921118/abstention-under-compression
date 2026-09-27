#!/usr/bin/env python3
"""Validate fake quant vs real bnb NF4 using the same pipeline.

Uses identical data loading, generation, and metric computation as
kaq_phase2.py, but loads the model with BitsAndBytesConfig instead of
applying fake quantization. This validates that fake quant NF4 results
are representative of real deployment.

Usage:
    CUDA_VISIBLE_DEVICES=5 python experiments/kaq_real_quant_validation.py \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --n-samples 2000 --seeds 5
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Import from kaq_phase2
_kaq_p2 = _ROOT / "experiments" / "kaq_phase2.py"
spec = importlib.util.spec_from_file_location("kaq_phase2", _kaq_p2)
kaq_phase2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kaq_phase2)

load_samples = kaq_phase2.load_samples
run_generation = kaq_phase2.run_generation
compute_metrics = kaq_phase2.compute_metrics
release_model = kaq_phase2.release_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--n-samples", type=int, default=2000)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--out-dir", default="results/kaq_real_quant_validation")
    return p.parse_args()


def load_model_bnb_nf4(model_id: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    print(f"  Loading {model_id} with bnb NF4...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="auto",
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Loaded in {time.time()-t0:.1f}s")
    return model, tokenizer


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Real Quant Validation")
    print(f"  Model: {args.model}")
    print(f"  N={args.n_samples} x {args.seeds} seeds\n")

    all_metrics = {}

    for seed in range(args.seeds):
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        print(f"{'='*60}")

        samples = load_samples(args.n_samples, seed=seed)
        tag = f"bnb_nf4_seed{seed}"
        raw_path = out_dir / f"{tag}.jsonl"

        if raw_path.exists() and raw_path.stat().st_size > 0:
            print(f"  [bnb_nf4] cached: {raw_path}")
            records = [json.loads(l)
                       for l in raw_path.read_text().splitlines() if l.strip()]
        else:
            model, tokenizer = load_model_bnb_nf4(args.model)

            print(f"  Running generation ({len(samples)} samples)...")
            t0 = time.time()
            records = run_generation(
                model, tokenizer, samples,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
            )
            elapsed = time.time() - t0
            print(f"  Done in {elapsed:.1f}s")

            with open(raw_path, "w") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

            release_model(model)

        metrics = compute_metrics(records)
        all_metrics[tag] = metrics
        print(f"  [bnb_nf4] acc={metrics['accuracy_on_answerable']:.3f}  "
              f"ECE={metrics['ece_10bin']:.4f}  "
              f"overconf={metrics['overconfidence_rate_tau0.8']:.3f}")

    # Aggregate
    print(f"\n\n{'='*60}")
    print(f"AGGREGATE ({args.seeds} seeds)")
    print(f"{'='*60}\n")

    seed_metrics = [all_metrics[f"bnb_nf4_seed{s}"] for s in range(args.seeds)]
    agg = {}
    for key in seed_metrics[0]:
        vals = [m[key] for m in seed_metrics if m[key] is not None]
        if vals and isinstance(vals[0], (int, float)):
            agg[key] = {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals, ddof=1)), 4) if len(vals) > 1 else 0,
            }

    print(f"  bnb_nf4: "
          f"acc={agg['accuracy_on_answerable']['mean']:.3f} "
          f"(+/-{agg['accuracy_on_answerable']['std']:.3f})  "
          f"ECE={agg['ece_10bin']['mean']:.4f}  "
          f"overconf={agg['overconfidence_rate_tau0.8']['mean']:.3f}")

    # Load fake quant results for comparison
    fake_path = _ROOT / "results" / "kaq_phase2" / "summary.json"
    if fake_path.exists():
        fake = json.loads(fake_path.read_text())
        fake_nf4 = fake["aggregate"]["nf4"]
        fake_fp16 = fake["aggregate"]["fp16"]

        print(f"\n  Comparison (same pipeline, same seeds):")
        print(f"  {'':15s}  {'acc':>8s}  {'overconf':>10s}  {'ECE':>8s}")
        print(f"  {'FP16 (fake)':15s}  "
              f"{fake_fp16['accuracy_on_answerable']['mean']:8.3f}  "
              f"{fake_fp16['overconfidence_rate_tau0.8']['mean']:10.3f}  "
              f"{fake_fp16['ece_10bin']['mean']:8.4f}")
        print(f"  {'NF4 fake':15s}  "
              f"{fake_nf4['accuracy_on_answerable']['mean']:8.3f}  "
              f"{fake_nf4['overconfidence_rate_tau0.8']['mean']:10.3f}  "
              f"{fake_nf4['ece_10bin']['mean']:8.4f}")
        print(f"  {'NF4 bnb (real)':15s}  "
              f"{agg['accuracy_on_answerable']['mean']:8.3f}  "
              f"{agg['overconfidence_rate_tau0.8']['mean']:10.3f}  "
              f"{agg['ece_10bin']['mean']:8.4f}")

        # Compute gap
        for metric_name, metric_key in [
            ("acc", "accuracy_on_answerable"),
            ("overconf", "overconfidence_rate_tau0.8"),
            ("ECE", "ece_10bin"),
        ]:
            fake_val = fake_nf4[metric_key]["mean"]
            real_val = agg[metric_key]["mean"]
            gap = real_val - fake_val
            print(f"  Gap (real-fake) {metric_name}: {gap:+.4f}")

    out = {
        "model": args.model,
        "method": "bnb_nf4",
        "n_samples": args.n_samples,
        "n_seeds": args.seeds,
        "per_seed": all_metrics,
        "aggregate": {"bnb_nf4": agg},
    }
    out_path = out_dir / "validation_results.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
