#!/usr/bin/env python3
"""Exp 6: Group size — how quantization granularity affects calibration.

Controlled variable: group_size ∈ {32, 64, 128, 256, channel-wise}.
Fixed: RTN rounding, uniform 4-bit integer grid, absmax scaling, no calibration data.

Measures:
  - Per-layer weight-space NMSE for each group size
  - Downstream MMLU: ECE, accuracy, overconf, AUROC, Brier

Usage (WSL2, needs GPU):
  python experiments/exp6_group_size.py
  python experiments/exp6_group_size.py --smoke
"""
from __future__ import annotations

import argparse, gc, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJ_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj",
              "gate_proj", "up_proj", "down_proj")

GS_VALUES = [32, 64, 128, 256, -1]  # -1 = channel-wise (gs=d_in)
GS_LABELS = ["gs32", "gs64", "gs128", "gs256", "channel"]


def fake_quant_uniform(W: torch.Tensor, bits: int = 4, gs: int = 128
                       ) -> torch.Tensor:
    W = W.float()
    d_out, d_in = W.shape
    qmax = 2 ** (bits - 1) - 1
    qmin = -(qmax + 1)
    g = gs if gs > 0 and d_in % gs == 0 else d_in
    Wg = W.reshape(d_out, d_in // g, g)
    scale = Wg.abs().amax(dim=2, keepdim=True) / qmax
    scale = scale.clamp(min=1e-12)
    Wq = (torch.clamp(torch.round(Wg / scale), qmin, qmax) * scale)
    return Wq.reshape(d_out, d_in)


def auroc(score, label):
    score = np.asarray(score, float)
    label = np.asarray(label, int)
    n_pos, n_neg = int(label.sum()), int((label == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    _, inv, cnt = np.unique(score, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt)
    ranks = ((csum - cnt + csum + 1) / 2.0)[inv]
    return float((ranks[label == 1].sum() - n_pos * (n_pos + 1) / 2.0)
                 / (n_pos * n_neg))


def calib_metrics(conf, correct, tau=0.8, n_bins=15):
    conf = np.asarray(conf, float)
    correct = np.asarray(correct, float)
    n = len(conf)
    overconf = float(((conf > tau) & (correct == 0)).mean())
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        m = (conf > bins[b]) & (conf <= bins[b + 1])
        if m.sum() > 0:
            ece += abs(correct[m].mean() - conf[m].mean()) * m.sum() / n
    return {"acc": float(correct.mean()), "overconf": overconf,
            "ece": float(ece),
            "brier": float(np.mean((conf - correct) ** 2)),
            "auroc": auroc(conf, correct)}


def iter_target_modules(layers):
    for blk_idx, block in enumerate(layers):
        for mod_name, mod in block.named_modules():
            if isinstance(mod, nn.Linear):
                leaf = mod_name.split(".")[-1]
                if leaf in PROJ_NAMES:
                    yield blk_idx, mod_name, leaf, mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--eval-n", type=int, default=1500)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    if args.smoke:
        args.eval_n = 60
        args.seeds = 1

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev == "cuda" else torch.float32
    print(f"[init] model={args.model}  bits={args.bits}")
    print(f"[init] group sizes: {GS_LABELS}")
    print(f"[init] device={dev}  eval_n={args.eval_n}  seeds={args.seeds}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype).to(dev).eval()
    layers = model.model.layers
    n_blocks = len(layers)
    n_targets = sum(1 for _ in iter_target_modules(layers))
    print(f"[init] {n_blocks} blocks, {n_targets} target layers")

    # ── Per-layer weight-space NMSE for each group size ──
    print(f"\n{'='*60}\nPer-layer weight-space NMSE\n{'='*60}")
    nmse_by_gs = {}
    for gs_val, gs_label in zip(GS_VALUES, GS_LABELS):
        layer_data = []
        for blk_idx, mod_name, leaf, mod in iter_target_modules(layers):
            W = mod.weight.data.float().cpu()
            W_q = fake_quant_uniform(W, args.bits, gs_val)
            w_norm = float((W ** 2).sum())
            nmse = float(((W - W_q) ** 2).sum()) / w_norm if w_norm > 0 else 0
            module_type = "attention" if leaf in ("q_proj","k_proj","v_proj","o_proj") else "mlp"
            layer_data.append({
                "name": f"block{blk_idx}.{leaf}",
                "block_idx": blk_idx, "proj_type": leaf,
                "module_type": module_type,
                "shape": list(W.shape),
                "nmse": round(nmse, 8),
            })
        nmse_vals = [d["nmse"] for d in layer_data]
        nmse_by_gs[gs_label] = {
            "per_layer": layer_data,
            "mean": round(float(np.mean(nmse_vals)), 8),
            "median": round(float(np.median(nmse_vals)), 8),
        }
        print(f"  {gs_label:8s}  NMSE mean={np.mean(nmse_vals):.6f}  "
              f"median={np.median(nmse_vals):.6f}")

    # ── MMLU setup ──
    mmlu = load_dataset("cais/mmlu", "all", split="test")
    opt_ids = [tok(" " + x, add_special_tokens=False).input_ids[-1]
               for x in ["A", "B", "C", "D"]]
    optt = torch.tensor(opt_ids, device=dev)

    def fmt(e):
        c = e["choices"]
        return (f"Question: {e['question']}\n"
                f"A. {c[0]}\nB. {c[1]}\nC. {c[2]}\nD. {c[3]}\nAnswer:")

    def encode(seed, n):
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(mmlu))[:n]
        exs = [mmlu[int(i)] for i in idx]
        batches = []
        for s in range(0, len(exs), args.batch):
            b = exs[s:s + args.batch]
            enc = tok([fmt(e) for e in b], return_tensors="pt",
                      padding=True).to(dev)
            ans = torch.tensor([e["answer"] for e in b], device=dev)
            batches.append((enc, ans))
        return batches

    @torch.no_grad()
    def evaluate(batches):
        conf_all, corr_all = [], []
        for enc, ans in batches:
            lg = model(**enc).logits[:, -1, :].index_select(1, optt)
            p = torch.softmax(lg.float(), dim=-1)
            mx, pred = p.max(dim=-1)
            conf_all.append(mx.cpu().numpy())
            corr_all.append((pred == ans).cpu().numpy().astype(int))
        if dev == "cuda":
            torch.cuda.synchronize()
        return np.concatenate(conf_all), np.concatenate(corr_all)

    def eval_multi_seed(label):
        all_m = []
        for si in range(args.seeds):
            batches = encode(1000 + si, args.eval_n)
            m = calib_metrics(*evaluate(batches), tau=args.tau)
            all_m.append(m)
            print(f"  [{label}] seed {si}: acc={m['acc']:.4f}  "
                  f"ece={m['ece']:.4f}  overconf={m['overconf']:.4f}")
            del batches
        avg = {}
        for key in all_m[0]:
            vals = [m[key] for m in all_m]
            avg[key] = round(float(np.mean(vals)), 6)
            avg[key + "_std"] = round(float(np.std(vals)), 6)
        return avg, all_m

    def apply_quant(gs_val, label):
        originals = {}
        for blk_idx, mod_name, leaf, mod in iter_target_modules(layers):
            originals[(blk_idx, mod_name)] = (mod, mod.weight.data.cpu())
            Wq = fake_quant_uniform(mod.weight.data.cpu(), args.bits, gs_val)
            mod.weight.data = Wq.to(dtype).to(dev)
        print(f"  {label} applied to {len(originals)} layers")
        return originals

    def restore(originals):
        for (_, _), (mod, w_orig) in originals.items():
            mod.weight.data = w_orig.to(dtype).to(dev)

    t_total = time.time()
    keys = ["acc", "ece", "overconf", "brier", "auroc"]

    # ── Phase 0: Baseline ──
    print(f"\n{'='*60}\nPHASE 0: Baseline (fp16)\n{'='*60}")
    t0 = time.time()
    baseline_avg, baseline_seeds = eval_multi_seed("fp16")
    t0 = time.time() - t0
    print(f"  done ({t0:.0f}s)")

    # ── Phase 1..5: Each group size ──
    condition_results = {}
    timings = {"baseline": round(t0, 1)}

    for gs_val, gs_label in zip(GS_VALUES, GS_LABELS):
        print(f"\n{'='*60}\nPHASE: {gs_label} ({args.bits}-bit)\n{'='*60}")
        t_c = time.time()
        orig = apply_quant(gs_val, gs_label)
        avg, seeds = eval_multi_seed(gs_label)
        restore(orig); del orig
        gc.collect()
        if dev == "cuda": torch.cuda.empty_cache()
        t_c = time.time() - t_c
        condition_results[gs_label] = {"avg": avg, "per_seed": seeds}
        timings[gs_label] = round(t_c, 1)
        print(f"  done ({t_c:.0f}s)")

    t_total = time.time() - t_total
    timings["total"] = round(t_total, 1)

    # ── Summary ──
    print(f"\n{'='*72}")
    print(f"RESULTS — Group size sweep: {GS_LABELS}")
    print(f"{'='*72}")
    print(f"Model: {args.model}  |  {args.bits}-bit uniform RTN  |  "
          f"N={args.eval_n} x {args.seeds} seeds")

    header = f"{'Metric':<12} {'fp16':>9}"
    for gl in GS_LABELS:
        header += f" {gl:>9}"
    print(f"\n{header}")
    print("-" * (14 + 10 * (1 + len(GS_LABELS))))

    for k in keys:
        row = f"{k:<12} {baseline_avg[k]:9.4f}"
        for gl in GS_LABELS:
            row += f" {condition_results[gl]['avg'][k]:9.4f}"
        print(row)

    print(f"\n  Deltas vs fp16:")
    for k in keys:
        row = f"  Δ {k:<9}"
        for gl in GS_LABELS:
            d = condition_results[gl]["avg"][k] - baseline_avg[k]
            row += f" {d:+9.4f}"
        print(row)

    print(f"\n  Weight-space NMSE (mean across 196 layers):")
    for gl in GS_LABELS:
        print(f"    {gl:8s}  {nmse_by_gs[gl]['mean']:.6f}")

    # gs128 as reference for rescue ratios
    ref = "gs128"
    ref_avg = condition_results[ref]["avg"]
    print(f"\n  Rescue ratio vs {ref} (how much damage each gs avoids relative to gs128):")
    for k in keys:
        row = f"  {k:<12}"
        ref_dmg = ref_avg[k] - baseline_avg[k]
        for gl in GS_LABELS:
            dmg = condition_results[gl]["avg"][k] - baseline_avg[k]
            if abs(ref_dmg) > 1e-6:
                ratio = 1.0 - dmg / ref_dmg
                row += f" {ratio:+9.3f}"
            else:
                row += f" {'n/a':>9}"
        print(row)

    # ── Save ──
    model_slug = args.model.split("/")[-1].lower().replace("-", "_")
    out_path = Path(args.output or
                    f"results/exp6_group_size_{model_slug}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    deltas = {}
    for gl in GS_LABELS:
        deltas[gl + "_vs_fp16"] = {
            k: round(condition_results[gl]["avg"][k] - baseline_avg[k], 6)
            for k in keys}

    out_data = {
        "experiment": "exp6_group_size",
        "model": args.model,
        "config": {"bits": args.bits, "group_sizes": GS_VALUES,
                   "group_size_labels": GS_LABELS,
                   "eval_n": args.eval_n, "seeds": args.seeds, "tau": args.tau},
        "results": {
            "baseline": {"avg": baseline_avg, "per_seed": baseline_seeds},
            **condition_results,
        },
        "deltas": deltas,
        "weight_nmse": {gl: {"mean": nmse_by_gs[gl]["mean"],
                             "median": nmse_by_gs[gl]["median"],
                             "per_layer": nmse_by_gs[gl]["per_layer"]}
                        for gl in GS_LABELS},
        "timing_s": timings,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
