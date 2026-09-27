#!/usr/bin/env python3
"""Exp 1: Codebook shape — uniform integer grid vs NF4 quantile codebook.

Controlled variable: codebook shape (uniform vs NF4 non-uniform).
Fixed: RTN rounding, group_size=128, absmax scaling, no calibration data.

Measures:
  - Per-layer weight-space NMSE: ‖W − W̃‖² / ‖W‖²
  - Downstream MMLU: ECE, accuracy, overconf, AUROC, Brier

Usage (WSL2, needs GPU):
  python experiments/exp1_codebook_shape.py
  python experiments/exp1_codebook_shape.py --smoke
"""
from __future__ import annotations

import argparse, gc, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJ_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj",
              "gate_proj", "up_proj", "down_proj")

# bitsandbytes NF4 codebook — 16 quantiles of N(0,1), normalized to [-1, 1]
NF4_CODEBOOK = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0])

NF4_MIDPOINTS = ((NF4_CODEBOOK[:-1] + NF4_CODEBOOK[1:]) / 2)


# ── quantization primitives ─────────────────────────────────────────────

def fake_quant_uniform(W: torch.Tensor, bits: int = 4, gs: int = 128
                       ) -> torch.Tensor:
    """Group-wise symmetric RTN on uniform integer grid (15 effective levels)."""
    W = W.float()
    d_out, d_in = W.shape
    qmax = 2 ** (bits - 1) - 1
    qmin = -(qmax + 1)
    g = gs if d_in % gs == 0 else d_in
    Wg = W.reshape(d_out, d_in // g, g)
    scale = Wg.abs().amax(dim=2, keepdim=True) / qmax
    scale = scale.clamp(min=1e-12)
    Wq = (torch.clamp(torch.round(Wg / scale), qmin, qmax) * scale)
    return Wq.reshape(d_out, d_in)


def fake_quant_nf4(W: torch.Tensor, gs: int = 128) -> torch.Tensor:
    """Group-wise NF4 codebook quantization (16 non-uniform levels).

    Same scale method (absmax) and group structure as uniform, only the
    quantization grid differs: N(0,1) quantiles instead of uniform integers.
    """
    W = W.float()
    d_out, d_in = W.shape
    g = gs if d_in % gs == 0 else d_in
    Wg = W.reshape(d_out, d_in // g, g)
    scale = Wg.abs().amax(dim=2, keepdim=True)
    scale = scale.clamp(min=1e-12)
    Wn = Wg / scale
    midpoints = NF4_MIDPOINTS.to(Wn.device)
    codebook = NF4_CODEBOOK.to(Wn.device)
    idx = torch.bucketize(Wn, midpoints)
    Wq = codebook[idx] * scale
    return Wq.reshape(d_out, d_in)


# ── calibration metrics ─────────────────────────────────────────────────

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


# ── helpers ──────────────────────────────────────────────────────────────

def iter_target_modules(layers):
    for blk_idx, block in enumerate(layers):
        for mod_name, mod in block.named_modules():
            if isinstance(mod, nn.Linear):
                leaf = mod_name.split(".")[-1]
                if leaf in PROJ_NAMES:
                    yield blk_idx, mod_name, leaf, mod


# ── main ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--gs", type=int, default=128)
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
    print(f"[init] model={args.model}  bits={args.bits}  gs={args.gs}")
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

    # ── Per-layer weight-space NMSE ──
    print(f"\n{'='*60}\nPer-layer weight-space NMSE\n{'='*60}")
    layer_data = []
    for blk_idx, mod_name, leaf, mod in iter_target_modules(layers):
        W = mod.weight.data.float().cpu()
        W_uni = fake_quant_uniform(W, args.bits, args.gs)
        W_nf4 = fake_quant_nf4(W, args.gs)

        w_norm = float((W ** 2).sum())
        nmse_uni = float(((W - W_uni) ** 2).sum()) / w_norm if w_norm > 0 else 0
        nmse_nf4 = float(((W - W_nf4) ** 2).sum()) / w_norm if w_norm > 0 else 0
        ratio = nmse_nf4 / nmse_uni if nmse_uni > 0 else float("inf")

        module_type = "attention" if leaf in ("q_proj","k_proj","v_proj","o_proj") else "mlp"
        layer_data.append({
            "name": f"block{blk_idx}.{leaf}",
            "block_idx": blk_idx,
            "proj_type": leaf,
            "module_type": module_type,
            "shape": list(W.shape),
            "nmse_uniform": round(nmse_uni, 8),
            "nmse_nf4": round(nmse_nf4, 8),
            "ratio_nf4_vs_uniform": round(ratio, 6),
        })

    # Print summary
    for mtype in ["attention", "mlp"]:
        sub = [d for d in layer_data if d["module_type"] == mtype]
        if not sub:
            continue
        ratios = [d["ratio_nf4_vs_uniform"] for d in sub]
        print(f"  {mtype.upper()} ({len(sub)} layers):")
        print(f"    NF4/Uniform ratio: mean={np.mean(ratios):.4f}  "
              f"median={np.median(ratios):.4f}")
        pct = (1 - np.mean(ratios)) * 100
        if pct > 0:
            print(f"    → NF4 reduces weight-space error by {pct:.1f}%")
        else:
            print(f"    → NF4 increases weight-space error by {-pct:.1f}%")

    for ptype in PROJ_NAMES:
        sub = [d for d in layer_data if d["proj_type"] == ptype]
        if sub:
            ratios = [d["ratio_nf4_vs_uniform"] for d in sub]
            print(f"    {ptype:10s}  ratio={np.mean(ratios):.4f}")

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

    def apply_quant(quant_fn, label):
        originals = {}
        for blk_idx, mod_name, leaf, mod in iter_target_modules(layers):
            originals[(blk_idx, mod_name)] = (mod, mod.weight.data.cpu())
            Wq = quant_fn(mod.weight.data.cpu())
            mod.weight.data = Wq.to(dtype).to(dev)
        print(f"  {label} applied to {len(originals)} layers")
        return originals

    def restore(originals):
        for (_, _), (mod, w_orig) in originals.items():
            mod.weight.data = w_orig.to(dtype).to(dev)

    t_total = time.time()

    # ── Phase 1: Baseline ──
    print(f"\n{'='*60}\nPHASE 1: Baseline (fp16)\n{'='*60}")
    t1 = time.time()
    baseline_avg, baseline_seeds = eval_multi_seed("fp16")
    t1 = time.time() - t1
    print(f"  done ({t1:.0f}s)")

    # ── Phase 2: Uniform ──
    print(f"\n{'='*60}\nPHASE 2: Uniform {args.bits}-bit (gs={args.gs})\n{'='*60}")
    t2 = time.time()
    orig = apply_quant(
        lambda W: fake_quant_uniform(W, args.bits, args.gs), "Uniform")
    uni_avg, uni_seeds = eval_multi_seed("Uniform")
    restore(orig); del orig
    gc.collect()
    if dev == "cuda": torch.cuda.empty_cache()
    t2 = time.time() - t2
    print(f"  done ({t2:.0f}s)")

    # ── Phase 3: NF4 ──
    print(f"\n{'='*60}\nPHASE 3: NF4 (gs={args.gs})\n{'='*60}")
    t3 = time.time()
    orig = apply_quant(lambda W: fake_quant_nf4(W, args.gs), "NF4")
    nf4_avg, nf4_seeds = eval_multi_seed("NF4")
    restore(orig); del orig
    gc.collect()
    if dev == "cuda": torch.cuda.empty_cache()
    t3 = time.time() - t3
    print(f"  done ({t3:.0f}s)")

    # ── Summary ──
    t_total = time.time() - t_total
    keys = ["acc", "ece", "overconf", "brier", "auroc"]

    print(f"\n{'='*72}")
    print(f"RESULTS — Codebook shape: Uniform vs NF4")
    print(f"{'='*72}")
    print(f"Model: {args.model}  |  {args.bits}-bit gs={args.gs}  |  "
          f"N={args.eval_n} x {args.seeds} seeds")
    print(f"\n{'Metric':<12} {'fp16':>9} {'Uniform':>9} {'NF4':>9}  "
          f"{'Δ(Uni-fp16)':>12} {'Δ(NF4-fp16)':>13} {'Δ(NF4-Uni)':>12}")
    print("-" * 80)
    for k in keys:
        b, u, n = baseline_avg[k], uni_avg[k], nf4_avg[k]
        print(f"{k:<12} {b:9.4f} {u:9.4f} {n:9.4f}  "
              f"{u-b:+12.4f} {n-b:+13.4f} {n-u:+12.4f}")

    # Rescue ratio
    print(f"\nRescue ratio = (Uniform_damage - NF4_damage) / Uniform_damage")
    for k in keys:
        b, u, n = baseline_avg[k], uni_avg[k], nf4_avg[k]
        dmg = u - b
        if abs(dmg) > 1e-6:
            rescue = (u - n) / dmg
            print(f"  {k:<12} rescue = {rescue:.3f}  "
                  f"(damage={dmg:+.4f}, recovered={u-n:+.4f})")

    # Weight-space NMSE summary
    all_ratios = [d["ratio_nf4_vs_uniform"] for d in layer_data]
    print(f"\nWeight-space NMSE (NF4/Uniform ratio):")
    print(f"  mean={np.mean(all_ratios):.4f}  median={np.median(all_ratios):.4f}")
    pct = (1 - np.mean(all_ratios)) * 100
    print(f"  → NF4 {'reduces' if pct > 0 else 'increases'} weight error "
          f"by {abs(pct):.1f}%")

    # ── Save ──
    model_slug = args.model.split("/")[-1].lower().replace("-", "_")
    out_path = Path(args.output or
                    f"results/exp1_codebook_shape_{model_slug}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    deltas = {}
    for pair, (a, b_dict) in [("uniform_vs_fp16", (uni_avg, baseline_avg)),
                               ("nf4_vs_fp16", (nf4_avg, baseline_avg)),
                               ("nf4_vs_uniform", (nf4_avg, uni_avg))]:
        deltas[pair] = {k: round(a[k] - b_dict[k], 6) for k in keys}

    rescue_ratios = {}
    for k in keys:
        dmg = uni_avg[k] - baseline_avg[k]
        if abs(dmg) > 1e-6:
            rescue_ratios[k] = round((uni_avg[k] - nf4_avg[k]) / dmg, 6)

    out_data = {
        "experiment": "exp1_codebook_shape",
        "model": args.model,
        "config": {"bits": args.bits, "group_size": args.gs,
                   "eval_n": args.eval_n, "seeds": args.seeds, "tau": args.tau},
        "results": {
            "baseline": {"avg": baseline_avg, "per_seed": baseline_seeds},
            "uniform": {"avg": uni_avg, "per_seed": uni_seeds},
            "nf4": {"avg": nf4_avg, "per_seed": nf4_seeds},
        },
        "deltas": deltas,
        "rescue_ratios": rescue_ratios,
        "weight_nmse": {
            "per_layer": layer_data,
            "summary": {
                "ratio_mean": round(float(np.mean(all_ratios)), 6),
                "ratio_median": round(float(np.median(all_ratios)), 6),
            },
        },
        "timing_s": {"baseline": round(t1, 1), "uniform": round(t2, 1),
                     "nf4": round(t3, 1), "total": round(t_total, 1)},
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
