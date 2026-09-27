"""Fit kurtosis -> optimal clip percentile mapping and compare strategies.

Compares MSE of:
  1. absmax + NF4 (baseline)
  2. Fixed clip + NF4 (each of 5 percentiles)
  3. Oracle adaptive (per-group best percentile)
  4. kappa-mapped adaptive (fitted rule)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from kaq_phase1 import (
    NF4_CODEBOOK, CLIP_PERCENTILES, GROUP_SIZE,
    reshape_to_groups, kurtosis_batch, _nearest_mse, _clip_mse,
    TARGET_MODULES, detect_module_type,
)

CLIP_PCT_ARR = np.array(CLIP_PERCENTILES)


def main():
    import torch
    from safetensors.torch import load_file
    from huggingface_hub import snapshot_download

    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    model_dir = Path(snapshot_download(model_id, local_files_only=True))
    st_files = sorted(model_dir.glob("*.safetensors"))
    print(f"Model: {model_id}\n")

    all_k: list[np.ndarray] = []
    all_mse_nf4: list[np.ndarray] = []
    all_clip_mses: list[np.ndarray] = []  # (5, ng) per layer

    t0 = time.time()
    n_done = 0

    for sf in st_files:
        tensors = load_file(str(sf), device="cpu")
        for name in sorted(tensors):
            if detect_module_type(name) is None:
                continue
            W = tensors[name]
            if W.ndim != 2:
                continue
            groups = reshape_to_groups(W.float().numpy(), GROUP_SIZE)
            ng = len(groups)
            if ng == 0:
                continue

            k = kurtosis_batch(groups)
            m_nf4 = _nearest_mse(groups, NF4_CODEBOOK)
            clip_stack = np.stack(
                [_clip_mse(groups, NF4_CODEBOOK, p) for p in CLIP_PERCENTILES], 0)

            all_k.append(k)
            all_mse_nf4.append(m_nf4)
            all_clip_mses.append(clip_stack)

            n_done += 1
            if n_done % 28 == 0:
                print(f"  {n_done} layers [{time.time()-t0:.0f}s]")
        del tensors

    K = np.concatenate(all_k)
    M_nf4 = np.concatenate(all_mse_nf4)
    # (5, total_groups)
    CM = np.concatenate(all_clip_mses, axis=1)
    total = len(K)
    print(f"\nTotal: {total} groups, {n_done} layers, {time.time()-t0:.0f}s\n")

    baseline = float(M_nf4.mean())

    # ── 1. Fixed percentile strategies ─────────────────────────────
    print(f"{'='*65}")
    print(f"{'Strategy':>35s}  {'MSE':>12s}  {'vs baseline':>10s}")
    print(f"{'='*65}")
    print(f"{'absmax + NF4 (baseline)':>35s}  {baseline:.6e}  {'0.00%':>10s}")

    best_fixed_mse = baseline
    best_fixed_pct = None
    for i, p in enumerate(CLIP_PERCENTILES):
        m = float(CM[i].mean())
        imp = (baseline - m) / baseline * 100
        tag = ""
        if m < best_fixed_mse:
            best_fixed_mse = m
            best_fixed_pct = p
        print(f"{f'fixed clip {p}% + NF4':>35s}  {m:.6e}  {imp:+.2f}%")

    # ── 2. Oracle adaptive ─────────────────────────────────────────
    oracle = CM.min(axis=0)  # best of 5 per group
    oracle_mean = float(oracle.mean())
    oracle_imp = (baseline - oracle_mean) / baseline * 100
    print(f"{'oracle adaptive (per-group best)':>35s}  "
          f"{oracle_mean:.6e}  {oracle_imp:+.2f}%")

    # ── 3. Kappa-mapped strategies ─────────────────────────────────
    # Strategy A: simple threshold rules
    best_rule_mse = baseline
    best_rule_name = ""

    # Find best pct index per group
    best_idx = np.argmin(CM, axis=0)

    # Try various threshold-based rules
    rules = [
        ("k<0 -> 99.0, else 99.9",
         lambda k: np.where(k < 0, 2, 0)),  # idx: 2=99.0, 0=99.9
        ("k<0 -> 98.0, else 99.5",
         lambda k: np.where(k < 0, 3, 1)),
        ("k<0.5 -> 99.0, k<2 -> 99.5, else 99.9",
         lambda k: np.where(k < 0.5, 2, np.where(k < 2, 1, 0))),
        ("k<0 -> 98.0, k<1 -> 99.0, k<3 -> 99.5, else 99.9",
         lambda k: np.where(k < 0, 3, np.where(k < 1, 2, np.where(k < 3, 1, 0)))),
        ("k<-0.2 -> 98.0, k<0.5 -> 99.0, k<2 -> 99.5, else 99.9",
         lambda k: np.where(k < -0.2, 3, np.where(k < 0.5, 2, np.where(k < 2, 1, 0)))),
    ]

    print(f"\n{'--- kappa-mapped rules ---':>35s}")
    for name, rule_fn in rules:
        idx = rule_fn(K).astype(int)
        mse_arr = CM[idx, np.arange(total)]
        m = float(mse_arr.mean())
        imp = (baseline - m) / baseline * 100
        tag = ""
        if m < best_rule_mse:
            best_rule_mse = m
            best_rule_name = name
        print(f"{name:>50s}  {m:.6e}  {imp:+.2f}%")

    # ── 4. Optimized bin-based rule ────────────────────────────────
    # For each kurtosis bin, pick the fixed percentile that minimizes
    # mean MSE within that bin
    print(f"\n{'--- optimized bin rule ---':>35s}")
    edges = [-999, -0.5, 0, 0.5, 1, 2, 3, 5, 999]
    opt_idx = np.zeros(total, dtype=int)
    for b in range(len(edges) - 1):
        lo, hi = edges[b], edges[b + 1]
        mask = (K >= lo) & (K < hi)
        if mask.sum() == 0:
            continue
        bin_means = CM[:, mask].mean(axis=1)
        best_i = int(np.argmin(bin_means))
        opt_idx[mask] = best_i
        pct = CLIP_PERCENTILES[best_i]
        n = int(mask.sum())
        print(f"    k in [{lo:>5},{hi:>4}): best={pct}%  n={n}")

    opt_mse_arr = CM[opt_idx, np.arange(total)]
    opt_m = float(opt_mse_arr.mean())
    opt_imp = (baseline - opt_m) / baseline * 100
    print(f"{'optimized 8-bin rule':>35s}  {opt_m:.6e}  {opt_imp:+.2f}%")

    # ── Summary ────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("Summary:")
    print(f"  Baseline (absmax+NF4):     {baseline:.6e}")
    print(f"  Best fixed clip ({best_fixed_pct}%): {best_fixed_mse:.6e}  "
          f"({(baseline-best_fixed_mse)/baseline*100:+.2f}%)")
    print(f"  Best simple rule:          {best_rule_mse:.6e}  "
          f"({(baseline-best_rule_mse)/baseline*100:+.2f}%)")
    print(f"  Optimized bin rule:        {opt_m:.6e}  ({opt_imp:+.2f}%)")
    print(f"  Oracle adaptive:           {oracle_mean:.6e}  ({oracle_imp:+.2f}%)")
    gap_closed = (opt_imp / oracle_imp * 100) if oracle_imp > 0 else 0
    print(f"\n  Bin rule closes {gap_closed:.1f}% of the gap to oracle.")
    print(f"{'='*65}")

    # ── Save ───────────────────────────────────────────────────────
    out = {
        "model": model_id,
        "n_groups": total,
        "baseline_mse": baseline,
        "fixed_clip_mse": {str(p): float(CM[i].mean())
                           for i, p in enumerate(CLIP_PERCENTILES)},
        "oracle_mse": oracle_mean,
        "oracle_improvement_pct": round(oracle_imp, 4),
        "best_fixed": {"pct": best_fixed_pct,
                       "mse": best_fixed_mse},
        "optimized_bin_rule": {
            "edges": edges,
            "mse": opt_m,
            "improvement_pct": round(opt_imp, 4),
            "gap_closed_pct": round(gap_closed, 1),
        },
    }
    out_path = Path("results") / "kaq_adaptive_fit.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
