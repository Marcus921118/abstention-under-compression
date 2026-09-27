"""Verify: does kurtosis predict the optimal clip percentile for NF4?

For each group (gs=128), finds which clip percentile minimizes MSE
when using NF4 codebook, then computes Spearman correlation with kurtosis.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

# reuse core functions from kaq_phase1
from kaq_phase1 import (
    NF4_CODEBOOK, FP4_CODEBOOK, CLIP_PERCENTILES, GROUP_SIZE, BATCH,
    reshape_to_groups, kurtosis_batch, _nearest_mse, _clip_mse,
    TARGET_MODULES, detect_module_type,
)

CLIP_PCT_ARR = np.array(CLIP_PERCENTILES)


def main():
    import torch
    from safetensors.torch import load_file
    from huggingface_hub import snapshot_download
    from scipy.stats import spearmanr

    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    model_dir = Path(snapshot_download(model_id, local_files_only=True))
    st_files = sorted(model_dir.glob("*.safetensors"))
    print(f"Model: {model_id}\n")

    mod_types_list = sorted(TARGET_MODULES)
    mod_idx = {m: i for i, m in enumerate(mod_types_list)}

    all_k: list[np.ndarray] = []
    all_bp: list[np.ndarray] = []
    all_mod: list[np.ndarray] = []

    t0 = time.time()
    n_done = 0

    for sf in st_files:
        tensors = load_file(str(sf), device="cpu")
        for name in sorted(tensors):
            mt = detect_module_type(name)
            if mt is None:
                continue
            W = tensors[name]
            if W.ndim != 2:
                continue

            groups = reshape_to_groups(W.float().numpy(), GROUP_SIZE)
            ng = len(groups)
            if ng == 0:
                continue

            k = kurtosis_batch(groups)
            cnf4 = np.stack(
                [_clip_mse(groups, NF4_CODEBOOK, p) for p in CLIP_PERCENTILES], 0)
            bi = np.argmin(cnf4, axis=0)
            bp = CLIP_PCT_ARR[bi]

            all_k.append(k)
            all_bp.append(bp)
            all_mod.append(np.full(ng, mod_idx[mt], dtype=np.int8))

            n_done += 1
            elapsed = time.time() - t0
            if n_done % 14 == 0:
                print(f"  {n_done} layers done [{elapsed:.0f}s]")

        del tensors

    K = np.concatenate(all_k)
    BP = np.concatenate(all_bp)
    MOD = np.concatenate(all_mod)
    total = len(K)
    print(f"\nTotal: {total} groups, {n_done} layers, {time.time()-t0:.0f}s\n")

    # ── Global correlation ─────────────────────────────────────────
    rho_all, p_all = spearmanr(K, BP)
    print(f"{'='*60}")
    print(f"Global Spearman(kurtosis, best_clip_pct): "
          f"rho={rho_all:.4f}  p={p_all:.2e}")
    print(f"{'='*60}")

    # ── Per-module-type correlation ────────────────────────────────
    print("\nPer-module-type Spearman(kurtosis, best_clip_pct):")
    mod_corr = {}
    for mt in mod_types_list:
        mi = mod_idx[mt]
        mask = MOD == mi
        if mask.sum() < 10:
            continue
        rho, p = spearmanr(K[mask], BP[mask])
        n = int(mask.sum())
        print(f"  {mt:12s}: rho={rho:.4f}  p={p:.2e}  n={n}")
        mod_corr[mt] = {"rho": round(float(rho), 4), "p": float(p), "n": n}

    # ── Best pct distribution by kurtosis bin ──────────────────────
    bins = [(-999, 0), (0, 1), (1, 3), (3, 5), (5, 999)]
    print("\nBest clip pct distribution by kurtosis bin:")
    print(f"  {'bin':>12s}", end="")
    for p in CLIP_PERCENTILES:
        print(f"  {p:>6.1f}%", end="")
    print(f"  {'n':>8s}  {'mean_k':>7s}")

    bin_stats = []
    for lo, hi in bins:
        mask = (K >= lo) & (K < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        bp_masked = BP[mask]
        row = {"bin": f"[{lo},{hi})"}
        pct_dist = {}
        print(f"  {f'[{lo},{hi})':>12s}", end="")
        for p in CLIP_PERCENTILES:
            frac = float((bp_masked == p).mean())
            pct_dist[str(p)] = round(frac, 4)
            print(f"  {frac*100:6.1f}%", end="")
        mean_k = float(K[mask].mean())
        print(f"  {n:8d}  {mean_k:7.2f}")
        row.update({"n": n, "mean_kurtosis": round(mean_k, 3),
                     "pct_distribution": pct_dist})
        bin_stats.append(row)

    # ── Mean best pct by kurtosis bin ──────────────────────────────
    print("\nMean optimal clip percentile by kurtosis bin:")
    for lo, hi in bins:
        mask = (K >= lo) & (K < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        mean_pct = float(BP[mask].mean())
        print(f"  [{lo:>4},{hi:>4}): mean_pct={mean_pct:.2f}%  n={n}")

    # ── Save ───────────────────────────────────────────────────────
    output = {
        "model": model_id,
        "n_groups": total,
        "global_spearman": {"rho": round(float(rho_all), 4), "p": float(p_all)},
        "module_type_correlation": mod_corr,
        "kurtosis_bin_analysis": bin_stats,
    }
    out_path = Path("results") / "kaq_clip_pct_correlation.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
