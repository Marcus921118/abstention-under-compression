#!/usr/bin/env python3
"""Exp 7: Kurtosis-MSE relationship — test chi-squared prediction of NF4 error.

Hypothesis: NF4 assumes W ~ N(0,σ²), so MSE follows a scaled chi-squared.
Groups deviating from normality (high kurtosis) have predictably higher MSE.

At gs=128 the absmax scale is already near-optimal (alpha ≈ 0.96), so the
scale-prediction angle has limited practical value. The diagnostic angle
is stronger: kurtosis predicts WHICH groups/layers suffer most from NF4,
and these may correlate with overconfidence.

Calibrated formula (gs=128):
    alpha_optimal ≈ 0.9588 × (1 + kurtosis)^(-0.0173)

This script loads model weights (CPU only, no GPU inference) and measures:
  1. Per-group kurtosis, skewness
  2. Actual NF4 MSE with absmax scale
  3. Predicted optimal alpha from kurtosis formula (gs=128 calibration)
  4. Actual optimal alpha via grid search
  5. Correlations and breakdown by layer type
  6. Top-kurtosis layers / groups (diagnostic)

Usage (WSL2, CPU-only — no GPU needed):
  python experiments/exp7_kurtosis_mse.py
  python experiments/exp7_kurtosis_mse.py --model Qwen/Qwen2.5-3B-Instruct
  python experiments/exp7_kurtosis_mse.py --smoke
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import stats as sp_stats

PROJ_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj",
              "gate_proj", "up_proj", "down_proj")

NF4_CODEBOOK = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0])

NF4_MIDPOINTS = (NF4_CODEBOOK[:-1] + NF4_CODEBOOK[1:]) / 2


def _quantize_nf4(Wn: torch.Tensor) -> torch.Tensor:
    idx = torch.bucketize(Wn, NF4_MIDPOINTS)
    return NF4_CODEBOOK[idx]


def group_mse(Wg: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    Wn = Wg / scale
    Wq = _quantize_nf4(Wn) * scale
    return ((Wg - Wq) ** 2).mean(dim=-1)


def find_optimal_alpha(Wg: torch.Tensor, absmax: torch.Tensor,
                       n_steps: int = 80) -> torch.Tensor:
    best_mse = torch.full(Wg.shape[:-1], float("inf"))
    best_alpha = torch.ones(Wg.shape[:-1])

    for alpha in torch.linspace(0.2, 1.5, n_steps):
        s = absmax * float(alpha)
        mse = group_mse(Wg, s)
        better = mse < best_mse
        best_mse = torch.where(better, mse, best_mse)
        best_alpha = torch.where(better, torch.tensor(float(alpha)), best_alpha)

    return best_alpha, best_mse


def predict_alpha(kurtosis: np.ndarray) -> np.ndarray:
    return 0.9588 * (1 + np.abs(kurtosis)) ** (-0.0173)


def iter_target_modules(layers):
    for blk_idx, block in enumerate(layers):
        for mod_name, mod in block.named_modules():
            if isinstance(mod, nn.Linear):
                leaf = mod_name.split(".")[-1]
                if leaf in PROJ_NAMES:
                    yield blk_idx, mod_name, leaf, mod


def analyze_layer(W: torch.Tensor, gs: int, layer_name: str):
    W = W.float()
    d_out, d_in = W.shape
    g = gs if d_in % gs == 0 else d_in
    n_groups_per_row = d_in // g
    Wg = W.reshape(d_out, n_groups_per_row, g)

    absmax = Wg.abs().amax(dim=2, keepdim=True).clamp(min=1e-12)
    mse_absmax = group_mse(Wg, absmax)

    opt_alpha, mse_optimal = find_optimal_alpha(Wg, absmax)

    Wg_flat = Wg.reshape(-1, g).numpy()
    n_total = Wg_flat.shape[0]

    mu = Wg_flat.mean(axis=1, keepdims=True)
    centered = Wg_flat - mu
    var = (centered ** 2).mean(axis=1)
    std = np.sqrt(var).clip(min=1e-12)
    z = centered / std[:, None]
    kurt_flat = (z ** 4).mean(axis=1) - 3.0
    skew_flat = (z ** 3).mean(axis=1)

    mse_abs_flat = mse_absmax.numpy().ravel()
    mse_opt_flat = mse_optimal.numpy().ravel()
    opt_alpha_flat = opt_alpha.numpy().ravel()

    pred_alpha_flat = predict_alpha(kurt_flat)
    pred_scale_np = absmax.squeeze(-1).numpy().ravel() * pred_alpha_flat
    pred_scale_t = torch.from_numpy(pred_scale_np).float().reshape(-1, 1)
    Wg_flat = Wg.reshape(-1, g)
    mse_pred_flat = group_mse(Wg_flat, pred_scale_t).numpy()

    rho_kurt_mse, p_kurt_mse = sp_stats.spearmanr(kurt_flat, mse_abs_flat)
    rho_kurt_alpha, p_kurt_alpha = sp_stats.spearmanr(kurt_flat, opt_alpha_flat)
    rho_pred_opt, p_pred_opt = sp_stats.spearmanr(pred_alpha_flat, opt_alpha_flat)

    improvement_pred = 1 - mse_pred_flat / mse_abs_flat
    improvement_pred = np.where(np.isfinite(improvement_pred), improvement_pred, 0)
    improvement_opt = 1 - mse_opt_flat / mse_abs_flat
    improvement_opt = np.where(np.isfinite(improvement_opt), improvement_opt, 0)

    q25, q50, q75 = np.percentile(kurt_flat, [25, 50, 75])
    quartile_labels = np.digitize(kurt_flat, [q25, q50, q75])

    quartile_stats = []
    for q in range(4):
        mask = quartile_labels == q
        if mask.sum() == 0:
            continue
        quartile_stats.append({
            "quartile": q,
            "n_groups": int(mask.sum()),
            "kurt_range": [round(float(kurt_flat[mask].min()), 4),
                           round(float(kurt_flat[mask].max()), 4)],
            "mean_mse_absmax": round(float(mse_abs_flat[mask].mean()), 8),
            "mean_mse_predicted": round(float(mse_pred_flat[mask].mean()), 8),
            "mean_mse_optimal": round(float(mse_opt_flat[mask].mean()), 8),
            "mean_improvement_pred": round(float(improvement_pred[mask].mean()), 4),
            "mean_improvement_opt": round(float(improvement_opt[mask].mean()), 4),
            "mean_opt_alpha": round(float(opt_alpha_flat[mask].mean()), 4),
            "mean_pred_alpha": round(float(pred_alpha_flat[mask].mean()), 4),
        })

    return {
        "layer": layer_name,
        "shape": list(W.shape),
        "n_groups": n_total,
        "kurtosis": {
            "mean": round(float(kurt_flat.mean()), 4),
            "std": round(float(kurt_flat.std()), 4),
            "median": round(float(np.median(kurt_flat)), 4),
            "p95": round(float(np.percentile(kurt_flat, 95)), 4),
            "p99": round(float(np.percentile(kurt_flat, 99)), 4),
        },
        "skewness_mean": round(float(skew_flat.mean()), 4),
        "mse_absmax_mean": round(float(mse_abs_flat.mean()), 8),
        "mse_predicted_mean": round(float(mse_pred_flat.mean()), 8),
        "mse_optimal_mean": round(float(mse_opt_flat.mean()), 8),
        "improvement_predicted_mean": round(float(improvement_pred.mean()), 4),
        "improvement_optimal_mean": round(float(improvement_opt.mean()), 4),
        "correlation": {
            "kurtosis_vs_mse": {"rho": round(rho_kurt_mse, 4),
                                "p": round(float(p_kurt_mse), 6)},
            "kurtosis_vs_opt_alpha": {"rho": round(rho_kurt_alpha, 4),
                                     "p": round(float(p_kurt_alpha), 6)},
            "predicted_vs_optimal_alpha": {"rho": round(rho_pred_opt, 4),
                                          "p": round(float(p_pred_opt), 6)},
        },
        "quartile_breakdown": quartile_stats,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--gs", type=int, default=128)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    print(f"[init] model={args.model}  gs={args.gs}")
    print(f"[init] Loading model weights (CPU only)...")

    from transformers import AutoModelForCausalLM

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32,
        device_map="cpu")
    model.eval()
    layers = model.model.layers
    n_blocks = len(layers)
    t_load = time.time() - t0
    print(f"[init] Loaded {n_blocks} blocks in {t_load:.0f}s")

    targets = list(iter_target_modules(layers))
    if args.smoke:
        targets = targets[:3]
    n_targets = len(targets)
    print(f"[init] Analyzing {n_targets} target layers")

    attn_types = {"q_proj", "k_proj", "v_proj", "o_proj"}
    mlp_types = {"gate_proj", "up_proj", "down_proj"}

    all_results = []
    attn_results = []
    mlp_results = []

    for i, (blk_idx, mod_name, leaf, mod) in enumerate(targets):
        layer_name = f"block{blk_idx}.{leaf}"
        t_start = time.time()
        W = mod.weight.data.cpu()

        result = analyze_layer(W, args.gs, layer_name)
        result["block_idx"] = blk_idx
        result["proj_type"] = leaf
        result["module_type"] = "attention" if leaf in attn_types else "mlp"
        all_results.append(result)

        if leaf in attn_types:
            attn_results.append(result)
        else:
            mlp_results.append(result)

        elapsed = time.time() - t_start
        rho = result["correlation"]["kurtosis_vs_mse"]["rho"]
        imp = result["improvement_predicted_mean"]
        print(f"  [{i+1}/{n_targets}] {layer_name:25s}  "
              f"kurt={result['kurtosis']['mean']:6.2f}  "
              f"rho(kurt,mse)={rho:.3f}  "
              f"pred_improve={imp:.1%}  "
              f"({elapsed:.1f}s)")

    def aggregate(results, label):
        if not results:
            return None
        all_kurt = [r["kurtosis"]["mean"] for r in results]
        all_mse_abs = [r["mse_absmax_mean"] for r in results]
        all_mse_pred = [r["mse_predicted_mean"] for r in results]
        all_mse_opt = [r["mse_optimal_mean"] for r in results]
        all_imp_pred = [r["improvement_predicted_mean"] for r in results]
        all_imp_opt = [r["improvement_optimal_mean"] for r in results]
        all_rho = [r["correlation"]["kurtosis_vs_mse"]["rho"] for r in results]

        rho_cross, p_cross = sp_stats.spearmanr(all_kurt, all_mse_abs)

        return {
            "label": label,
            "n_layers": len(results),
            "kurtosis_mean": round(float(np.mean(all_kurt)), 4),
            "kurtosis_std": round(float(np.std(all_kurt)), 4),
            "mse_absmax_mean": round(float(np.mean(all_mse_abs)), 8),
            "mse_predicted_mean": round(float(np.mean(all_mse_pred)), 8),
            "mse_optimal_mean": round(float(np.mean(all_mse_opt)), 8),
            "improvement_predicted_mean": round(float(np.mean(all_imp_pred)), 4),
            "improvement_optimal_mean": round(float(np.mean(all_imp_opt)), 4),
            "mean_within_layer_rho": round(float(np.mean(all_rho)), 4),
            "cross_layer_rho": round(float(rho_cross), 4) if np.isfinite(rho_cross) else None,
            "cross_layer_p": round(float(p_cross), 6) if np.isfinite(p_cross) else None,
        }

    summary_all = aggregate(all_results, "all")
    summary_attn = aggregate(attn_results, "attention")
    summary_mlp = aggregate(mlp_results, "mlp")

    print(f"\n{'='*72}")
    print(f"SUMMARY — Kurtosis-MSE Analysis")
    print(f"{'='*72}")
    print(f"Model: {args.model}  |  gs={args.gs}")
    print(f"Prediction formula: alpha = 0.9588 * (1 + kurt)^(-0.0173)")

    for s in [summary_all, summary_attn, summary_mlp]:
        if s is None:
            continue
        print(f"\n  [{s['label']}] {s['n_layers']} layers")
        print(f"    Mean kurtosis:         {s['kurtosis_mean']:.3f} ± {s['kurtosis_std']:.3f}")
        print(f"    MSE (absmax):          {s['mse_absmax_mean']:.8f}")
        print(f"    MSE (predicted alpha):  {s['mse_predicted_mean']:.8f}")
        print(f"    MSE (optimal alpha):    {s['mse_optimal_mean']:.8f}")
        print(f"    Improvement (pred):     {s['improvement_predicted_mean']:.1%}")
        print(f"    Improvement (optimal):  {s['improvement_optimal_mean']:.1%}")
        print(f"    Within-layer rho(kurt,mse): {s['mean_within_layer_rho']:.3f}")
        if s['cross_layer_rho'] is not None:
            print(f"    Cross-layer rho(kurt,mse):  {s['cross_layer_rho']:.3f} (p={s['cross_layer_p']:.4f})")

    model_slug = args.model.split("/")[-1].lower().replace("-", "_")
    out_path = Path(args.output or
                    f"results/exp7_kurtosis_mse_{model_slug}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Diagnostic: top-10 layers by mean kurtosis
    sorted_by_kurt = sorted(all_results, key=lambda r: r["kurtosis"]["mean"],
                            reverse=True)
    print(f"\n  Top-10 layers by mean kurtosis:")
    print(f"  {'Layer':<28s} {'type':<6s} {'kurt':>8s} {'MSE_abs':>12s} {'p95_kurt':>10s}")
    for r in sorted_by_kurt[:10]:
        print(f"  {r['layer']:<28s} {r['module_type']:<6s} "
              f"{r['kurtosis']['mean']:>8.3f} {r['mse_absmax_mean']:>12.8f} "
              f"{r['kurtosis']['p95']:>10.3f}")

    # Diagnostic: kurtosis by block position
    block_kurt = {}
    for r in all_results:
        bi = r["block_idx"]
        if bi not in block_kurt:
            block_kurt[bi] = []
        block_kurt[bi].append(r["kurtosis"]["mean"])

    print(f"\n  Mean kurtosis by block position:")
    for bi in sorted(block_kurt.keys()):
        vals = block_kurt[bi]
        print(f"    block {bi:2d}: {np.mean(vals):.3f}")

    out_data = {
        "experiment": "exp7_kurtosis_mse",
        "model": args.model,
        "config": {"group_size": args.gs,
                   "prediction_formula": "alpha = 0.9588 * (1 + kurtosis)^(-0.0173)",
                   "note": "at gs=128 absmax is near-optimal; diagnostic value > predictive value"},
        "summary": {
            "all": summary_all,
            "attention": summary_attn,
            "mlp": summary_mlp,
        },
        "per_layer": all_results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")

    del model
    import gc
    gc.collect()


if __name__ == "__main__":
    main()
