"""Ablation: universal vs kappa-binned codebook, with and without clipping.

Answers:
  A. Is the gain from kappa-binning or from a universally better codebook?
  B. Do custom codebook + clip stack?

CPU-only. ~15 min.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from kaq_phase1 import (
    NF4_CODEBOOK, GROUP_SIZE,
    reshape_to_groups, kurtosis_batch, _nearest_mse, _clip_mse,
    TARGET_MODULES, detect_module_type,
)
from kaq_custom_codebook import kmeans_1d

KAPPA_EDGES = [-999, 0, 1, 3, 999]
KAPPA_LABELS = ["k<0", "0<=k<1", "1<=k<3", "k>=3"]
CLIP_PCTS = [99.7, 99.8, 99.85, 99.9, 99.95]
SUBSAMPLE_GROUPS = 8000
BATCH = 8000


def _clip_mse_cb(groups, codebook, pct):
    """clip + custom codebook MSE (same as _clip_mse but explicit)."""
    N = len(groups)
    mse = np.empty(N, dtype=np.float64)
    cb = codebook[None, None, :]
    for s in range(0, N, BATCH):
        e = min(s + BATCH, N)
        g = groups[s:e]
        cval = np.percentile(np.abs(g), pct, axis=1, keepdims=True)
        cval = np.maximum(cval, 1e-30)
        clipped = np.clip(g, -cval, cval)
        norm = clipped / cval
        idx = np.argmin(np.abs(norm[:, :, None] - cb), axis=2)
        deq = codebook[idx] * cval
        mse[s:e] = np.mean((g - deq) ** 2, axis=1)
    return mse


def load_layers(model_dir):
    from safetensors.torch import load_file
    for sf in sorted(model_dir.glob("*.safetensors")):
        tensors = load_file(str(sf), device="cpu")
        for name in sorted(tensors):
            if detect_module_type(name) is None:
                continue
            W = tensors[name]
            if W.ndim != 2:
                continue
            groups = reshape_to_groups(W.float().numpy(), GROUP_SIZE)
            if len(groups) > 0:
                yield name, groups
        del tensors


def main():
    from huggingface_hub import snapshot_download

    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    model_dir = Path(snapshot_download(model_id, local_files_only=True))
    n_bins = len(KAPPA_EDGES) - 1

    # Load existing per-bin codebooks
    cb_path = Path("results/kaq_custom_codebook.json")
    cb_data = json.loads(cb_path.read_text())
    bin_codebooks = [np.array(cb_data["codebooks"][lb]) for lb in KAPPA_LABELS]

    print(f"Model: {model_id}")
    print(f"Clip percentiles: {CLIP_PCTS}\n")

    # ── Pass 1: collect data + design universal codebook ──────────
    print("=== Pass 1: collect + design universal codebook ===")
    all_k, all_mse_nf4 = [], []
    uni_pool = []
    uni_sampled = 0
    rng = np.random.default_rng(42)
    t0 = time.time()
    n_done = 0

    for _name, groups in load_layers(model_dir):
        ng = len(groups)
        k = kurtosis_batch(groups)
        all_k.append(k)
        all_mse_nf4.append(_nearest_mse(groups, NF4_CODEBOOK))

        # subsample for universal codebook
        if uni_sampled < SUBSAMPLE_GROUPS:
            amax = np.max(np.abs(groups), axis=1, keepdims=True)
            amax = np.maximum(amax, 1e-30)
            normed = groups / amax
            take = min(ng, SUBSAMPLE_GROUPS - uni_sampled)
            if take < ng:
                idx = rng.choice(ng, take, replace=False)
                normed = normed[idx]
            uni_pool.append(normed.ravel())
            uni_sampled += take

        n_done += 1
        if n_done % 28 == 0:
            print(f"  {n_done} layers [{time.time()-t0:.0f}s]")

    K_all = np.concatenate(all_k)
    M_nf4 = np.concatenate(all_mse_nf4)
    total = len(K_all)
    baseline = float(M_nf4.mean())
    print(f"\nPass 1: {total} groups, {time.time()-t0:.0f}s")
    print(f"Baseline NF4: {baseline:.6e}\n")

    # Design universal codebook
    pooled = np.concatenate(uni_pool)
    print(f"Designing universal codebook ({len(pooled)} values)...")
    uni_cb = kmeans_1d(pooled, k=16)
    print(f"  NF4:       [{', '.join(f'{v:+.3f}' for v in NF4_CODEBOOK)}]")
    print(f"  Universal: [{', '.join(f'{v:+.3f}' for v in uni_cb)}]")
    del uni_pool, pooled

    # ── Pass 2: evaluate all strategies ───────────────────────────
    print(f"\n=== Pass 2: evaluate all strategies ===")
    # Strategies to evaluate:
    #   1. universal custom codebook (absmax)
    #   2. kappa-binned custom codebook (absmax) — verify cached result
    #   3. clip + NF4 (best pct)
    #   4. clip + universal custom
    #   5. clip + kappa-binned custom

    all_mse_uni = []
    all_mse_kbin = []
    # clip results: {pct: {strategy: [mse_arrays]}}
    clip_nf4 = {p: [] for p in CLIP_PCTS}
    clip_uni = {p: [] for p in CLIP_PCTS}
    clip_kbin = {p: [] for p in CLIP_PCTS}

    t1 = time.time()
    n_done = 0

    for _name, groups in load_layers(model_dir):
        ng = len(groups)
        k = kurtosis_batch(groups)

        # 1. Universal codebook
        all_mse_uni.append(_nearest_mse(groups, uni_cb))

        # 2. Kappa-binned codebook
        mse_kb = np.empty(ng, dtype=np.float64)
        for b in range(n_bins):
            lo, hi = KAPPA_EDGES[b], KAPPA_EDGES[b + 1]
            mask = (k >= lo) & (k < hi)
            if mask.any():
                mse_kb[mask] = _nearest_mse(groups[mask], bin_codebooks[b])
        all_mse_kbin.append(mse_kb)

        # 3-5. Clip + various codebooks
        for p in CLIP_PCTS:
            clip_nf4[p].append(_clip_mse(groups, NF4_CODEBOOK, p))
            clip_uni[p].append(_clip_mse_cb(groups, uni_cb, p))

            mse_ckb = np.empty(ng, dtype=np.float64)
            for b in range(n_bins):
                lo, hi = KAPPA_EDGES[b], KAPPA_EDGES[b + 1]
                mask = (k >= lo) & (k < hi)
                if mask.any():
                    mse_ckb[mask] = _clip_mse_cb(
                        groups[mask], bin_codebooks[b], p)
            clip_kbin[p].append(mse_ckb)

        n_done += 1
        if n_done % 14 == 0:
            print(f"  {n_done} layers [{time.time()-t1:.0f}s]")

    print(f"\nPass 2: {time.time()-t1:.0f}s\n")

    # ── Aggregate ─────────────────────────────────────────────────
    M_uni = np.concatenate(all_mse_uni)
    M_kbin = np.concatenate(all_mse_kbin)

    def imp(mse):
        return (baseline - mse) / baseline * 100

    sep = "=" * 74
    print(sep)
    print(f"{'Strategy':>50s}  {'MSE':>12s}  {'improve':>8s}")
    print(sep)

    rows = []

    def row(name, mse_val):
        i = imp(mse_val)
        print(f"{name:>50s}  {mse_val:.6e}  {i:+.2f}%")
        rows.append({"strategy": name, "mse": mse_val, "improvement_pct": round(i, 4)})

    row("absmax + NF4 (baseline)", baseline)
    row("absmax + universal custom", float(M_uni.mean()))
    row("absmax + kappa-binned custom (4 bins)", float(M_kbin.mean()))

    print()
    for p in CLIP_PCTS:
        cn = float(np.concatenate(clip_nf4[p]).mean())
        row(f"clip {p}% + NF4", cn)

    print()
    for p in CLIP_PCTS:
        cu = float(np.concatenate(clip_uni[p]).mean())
        row(f"clip {p}% + universal custom", cu)

    print()
    for p in CLIP_PCTS:
        ck = float(np.concatenate(clip_kbin[p]).mean())
        row(f"clip {p}% + kappa-binned custom", ck)

    print(sep)

    # ── Best combos summary ───────────────────────────────────────
    print("\n=== Best of each category ===")
    cats = {
        "NF4 only": baseline,
        "universal codebook": float(M_uni.mean()),
        "kappa-binned codebook": float(M_kbin.mean()),
    }
    for p in CLIP_PCTS:
        cn = float(np.concatenate(clip_nf4[p]).mean())
        if f"clip+NF4" not in cats or cn < cats["clip+NF4"]:
            cats["clip+NF4"] = cn
            cats["clip+NF4_pct"] = p
        cu = float(np.concatenate(clip_uni[p]).mean())
        if "clip+universal" not in cats or cu < cats["clip+universal"]:
            cats["clip+universal"] = cu
            cats["clip+universal_pct"] = p
        ck = float(np.concatenate(clip_kbin[p]).mean())
        if "clip+kappa-binned" not in cats or ck < cats["clip+kappa-binned"]:
            cats["clip+kappa-binned"] = ck
            cats["clip+kappa-binned_pct"] = p

    for k, v in cats.items():
        if k.endswith("_pct"):
            continue
        print(f"  {k:>30s}: {v:.6e}  ({imp(v):+.2f}%)")

    print(sep)

    # ── Save ──────────────────────────────────────────────────────
    out = {
        "model": model_id,
        "n_groups": total,
        "baseline_mse": baseline,
        "universal_codebook": uni_cb.tolist(),
        "results": rows,
        "best_per_category": {
            k: v for k, v in cats.items() if not k.endswith("_pct")
        },
    }
    out_path = Path("results") / "kaq_codebook_ablation.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
