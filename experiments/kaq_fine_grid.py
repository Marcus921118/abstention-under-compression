"""Fine-grid percentile sweep for kurtosis-adaptive clipping.

Tests 11 percentiles (99.0-99.99) instead of the coarse 5.
Goal: see if finer candidates let the kappa-bin rule close more
of the oracle gap.

CPU-only. ~25 min on 10M groups.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from kaq_phase1 import (
    NF4_CODEBOOK, GROUP_SIZE,
    reshape_to_groups, kurtosis_batch, _nearest_mse, _clip_mse,
    TARGET_MODULES, detect_module_type,
)

FINE_PCTS = [99.0, 99.2, 99.4, 99.5, 99.6, 99.7, 99.8, 99.85, 99.9, 99.95, 99.99]


def main():
    import torch
    from safetensors.torch import load_file
    from huggingface_hub import snapshot_download

    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    model_dir = Path(snapshot_download(model_id, local_files_only=True))
    st_files = sorted(model_dir.glob("*.safetensors"))
    print(f"Model: {model_id}")
    print(f"Percentiles: {FINE_PCTS}\n")

    all_k: list[np.ndarray] = []
    all_mse_nf4: list[np.ndarray] = []
    all_clip_mses: list[np.ndarray] = []

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
                [_clip_mse(groups, NF4_CODEBOOK, p) for p in FINE_PCTS], 0)

            all_k.append(k)
            all_mse_nf4.append(m_nf4)
            all_clip_mses.append(clip_stack)

            n_done += 1
            if n_done % 28 == 0:
                elapsed = time.time() - t0
                print(f"  {n_done} layers [{elapsed:.0f}s]")
        del tensors

    K = np.concatenate(all_k)
    M_nf4 = np.concatenate(all_mse_nf4)
    CM = np.concatenate(all_clip_mses, axis=1)  # (n_pcts, total_groups)
    total = len(K)
    print(f"\nTotal: {total} groups, {n_done} layers, {time.time()-t0:.0f}s\n")

    baseline = float(M_nf4.mean())

    # ── 1. Fixed percentile strategies ─────────────────────────────
    sep = "=" * 70
    print(sep)
    print(f"{'Strategy':>40s}  {'MSE':>12s}  {'vs baseline':>10s}")
    print(sep)
    print(f"{'absmax + NF4 (baseline)':>40s}  {baseline:.6e}  {'0.00%':>10s}")

    best_fixed_mse = baseline
    best_fixed_pct = None
    for i, p in enumerate(FINE_PCTS):
        m = float(CM[i].mean())
        imp = (baseline - m) / baseline * 100
        if m < best_fixed_mse:
            best_fixed_mse = m
            best_fixed_pct = p
        print(f"{f'fixed clip {p}% + NF4':>40s}  {m:.6e}  {imp:+.2f}%")

    # ── 2. Oracle adaptive ─────────────────────────────────────────
    oracle = CM.min(axis=0)
    oracle_mean = float(oracle.mean())
    oracle_imp = (baseline - oracle_mean) / baseline * 100
    print(f"{'oracle adaptive (per-group best)':>40s}  "
          f"{oracle_mean:.6e}  {oracle_imp:+.2f}%")

    # ── 3. Optimized bin rule ──────────────────────────────────────
    print(f"\n{'--- optimized bin rule ---':>40s}")
    edges = [-999, -0.5, 0, 0.5, 1, 2, 3, 5, 999]
    opt_idx = np.zeros(total, dtype=int)
    bin_details = []
    for b in range(len(edges) - 1):
        lo, hi = edges[b], edges[b + 1]
        mask = (K >= lo) & (K < hi)
        if mask.sum() == 0:
            continue
        bin_means = CM[:, mask].mean(axis=1)
        best_i = int(np.argmin(bin_means))
        opt_idx[mask] = best_i
        pct = FINE_PCTS[best_i]
        n = int(mask.sum())
        imp_vs_base = (baseline - float(bin_means[best_i])) / baseline * 100
        print(f"    k in [{lo:>5},{hi:>4}): best={pct}%  n={n}")
        bin_details.append({
            "bin": f"[{lo},{hi})", "best_pct": pct, "n": n,
            "all_pct_mses": {str(FINE_PCTS[j]): float(bin_means[j])
                             for j in range(len(FINE_PCTS))},
        })

    opt_mse_arr = CM[opt_idx, np.arange(total)]
    opt_m = float(opt_mse_arr.mean())
    opt_imp = (baseline - opt_m) / baseline * 100
    print(f"{'optimized 8-bin rule':>40s}  {opt_m:.6e}  {opt_imp:+.2f}%")

    # ── 4. Per-group best-pct distribution ─────────────────────────
    best_idx = np.argmin(CM, axis=0)
    print(f"\n{'--- per-group best pct distribution ---':>40s}")
    for i, p in enumerate(FINE_PCTS):
        frac = float((best_idx == i).mean()) * 100
        print(f"    {p:>6.2f}%: {frac:5.1f}% of groups")

    # ── Summary ────────────────────────────────────────────────────
    gap_closed = (opt_imp / oracle_imp * 100) if oracle_imp > 0 else 0
    print(f"\n{sep}")
    print("Summary:")
    print(f"  Baseline (absmax+NF4):     {baseline:.6e}")
    print(f"  Best fixed clip ({best_fixed_pct}%): {best_fixed_mse:.6e}  "
          f"({(baseline-best_fixed_mse)/baseline*100:+.2f}%)")
    print(f"  Optimized bin rule:        {opt_m:.6e}  ({opt_imp:+.2f}%)")
    print(f"  Oracle adaptive:           {oracle_mean:.6e}  ({oracle_imp:+.2f}%)")
    print(f"\n  Bin rule closes {gap_closed:.1f}% of the gap to oracle.")

    # Compare with coarse grid
    coarse_path = Path("results/kaq_adaptive_fit.json")
    if coarse_path.exists():
        coarse = json.loads(coarse_path.read_text())
        coarse_oracle = coarse["oracle_mse"]
        coarse_gap = coarse["optimized_bin_rule"]["gap_closed_pct"]
        print(f"\n  Coarse grid (5 pcts): oracle={coarse_oracle:.6e} "
              f"({coarse['oracle_improvement_pct']:+.2f}%), "
              f"bin rule closed {coarse_gap}%")
        print(f"  Fine grid ({len(FINE_PCTS)} pcts): oracle={oracle_mean:.6e} "
              f"({oracle_imp:+.2f}%), "
              f"bin rule closed {gap_closed:.1f}%")
    print(sep)

    # ── Save ───────────────────────────────────────────────────────
    out = {
        "model": model_id,
        "n_groups": total,
        "percentiles": FINE_PCTS,
        "baseline_mse": baseline,
        "fixed_clip_mse": {str(p): float(CM[i].mean())
                           for i, p in enumerate(FINE_PCTS)},
        "oracle_mse": oracle_mean,
        "oracle_improvement_pct": round(oracle_imp, 4),
        "best_fixed": {"pct": best_fixed_pct, "mse": best_fixed_mse},
        "optimized_bin_rule": {
            "edges": edges,
            "mse": opt_m,
            "improvement_pct": round(opt_imp, 4),
            "gap_closed_pct": round(gap_closed, 1),
            "bin_details": bin_details,
        },
        "per_group_best_distribution": {
            str(FINE_PCTS[i]): round(float((best_idx == i).mean()), 4)
            for i in range(len(FINE_PCTS))
        },
    }
    out_path = Path("results") / "kaq_fine_grid.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
