"""KAQ Phase 1: Weight-only kurtosis-adaptive quantization analysis.

Loads FP16 weights from a cached model, computes per-group (gs=128):
  - Excess kurtosis
  - NF4 / FP4 / clip+FP4 quantization MSE
Sweeps threshold tau for optimal KAQ cutpoint.

CPU-only. No GPU required.

Usage:
  python experiments/kaq_phase1.py                   # full run
  python experiments/kaq_phase1.py --test             # known-answer tests only
  python experiments/kaq_phase1.py --layers 2         # quick vertical slice
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# ── Codebooks (normalized to [-1, 1]) ──────────────────────────────

NF4_CODEBOOK = np.array([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0,
], dtype=np.float64)

_FP4_POS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]) / 6.0
FP4_CODEBOOK = np.sort(np.concatenate([-_FP4_POS[1:], _FP4_POS]))  # 15 values

CLIP_PERCENTILES = [99.9, 99.5, 99.0, 98.0, 95.0]
GROUP_SIZE = 128
BATCH = 8000

# ── Vectorized core ───────────────────────────────────────────────

def reshape_to_groups(W: np.ndarray, gs: int = GROUP_SIZE) -> np.ndarray:
    flat = W.reshape(-1).astype(np.float64)
    n = len(flat) // gs
    return flat[: n * gs].reshape(n, gs)


def kurtosis_batch(groups: np.ndarray) -> np.ndarray:
    m = groups.mean(axis=1, keepdims=True)
    d = groups - m
    m2 = (d ** 2).mean(axis=1)
    m4 = (d ** 4).mean(axis=1)
    safe = m2 > 1e-30
    out = np.zeros(len(groups), dtype=np.float64)
    out[safe] = m4[safe] / (m2[safe] ** 2) - 3.0
    return out


def _nearest_mse(groups: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    N = len(groups)
    mse = np.empty(N, dtype=np.float64)
    cb = codebook[None, None, :]
    for s in range(0, N, BATCH):
        e = min(s + BATCH, N)
        g = groups[s:e]
        amax = np.max(np.abs(g), axis=1, keepdims=True)
        amax = np.maximum(amax, 1e-30)
        norm = g / amax
        idx = np.argmin(np.abs(norm[:, :, None] - cb), axis=2)
        deq = codebook[idx] * amax
        mse[s:e] = np.mean((g - deq) ** 2, axis=1)
    return mse


def _clip_mse(groups: np.ndarray, codebook: np.ndarray, pct: float) -> np.ndarray:
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


# ── Tau sweep ──────────────────────────────────────────────────────

def sweep_tau(
    kappas: np.ndarray,
    mse_nf4: np.ndarray,
    mse_clip: np.ndarray,
    taus: np.ndarray,
) -> list[dict]:
    baseline = float(mse_nf4.mean())
    results = []
    for tau in taus:
        high = kappas >= tau
        n_hi = int(high.sum())
        n_lo = len(kappas) - n_hi
        frac = n_hi / len(kappas) if len(kappas) else 0.0
        kaq = np.where(high, mse_clip, mse_nf4)
        m = float(kaq.mean())
        imp = (baseline - m) / baseline * 100 if baseline > 0 else 0.0
        results.append({
            "tau": round(float(tau), 2),
            "n_low": n_lo,
            "n_high": n_hi,
            "frac_high": round(frac, 4),
            "mse_kaq": m,
            "improvement_pct": round(imp, 4),
        })
    return results


# ── Known-answer tests ─────────────────────────────────────────────

def run_tests():
    rng = np.random.default_rng(42)

    k_norm = kurtosis_batch(rng.normal(0, 1, (1, 10000)))[0]
    k_unif = kurtosis_batch(rng.uniform(-1, 1, (1, 10000)))[0]
    k_t3 = kurtosis_batch(rng.standard_t(3, (1, 10000)))[0]
    assert abs(k_norm) < 0.2, f"normal kurtosis: {k_norm}"
    assert abs(k_unif - (-1.2)) < 0.2, f"uniform kurtosis: {k_unif}"
    assert k_t3 > 2, f"t(3) kurtosis: {k_t3}"

    w_exact = np.tile(NF4_CODEBOOK, GROUP_SIZE // 16 + 1)[:GROUP_SIZE]
    w_exact = (w_exact * 2.5).reshape(1, -1)
    mse = _nearest_mse(w_exact, NF4_CODEBOOK)[0]
    assert mse < 1e-20, f"exact MSE: {mse}"

    assert len(NF4_CODEBOOK) == 16
    assert len(FP4_CODEBOOK) == 15

    heavy = rng.standard_t(2.5, (1, GROUP_SIZE))
    cm = _clip_mse(heavy, FP4_CODEBOOK, 99.0)[0]
    assert cm >= 0

    print("Tests passed:")
    print(f"  kurtosis normal={k_norm:.3f}  uniform={k_unif:.3f}  t3={k_t3:.3f}")
    print(f"  exact NF4 MSE={mse:.1e}")
    print()


# ── Main ───────────────────────────────────────────────────────────

TARGET_MODULES = {"q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"}


def detect_module_type(name: str) -> str | None:
    for k in TARGET_MODULES:
        if k in name:
            return k
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--layers", type=int, default=0,
                        help="First N weight tensors (0=all)")
    args = parser.parse_args()

    run_tests()
    if args.test:
        return

    import torch
    from safetensors.torch import load_file
    from huggingface_hub import snapshot_download

    model_dir = Path(snapshot_download(args.model, local_files_only=True))
    st_files = sorted(model_dir.glob("*.safetensors"))
    if not st_files:
        sys.exit(f"No safetensors in {model_dir}")
    print(f"Model: {args.model}")
    print(f"Dir:   {model_dir}")
    print(f"Files: {len(st_files)}\n")

    all_kappas: list[np.ndarray] = []
    all_mse_nf4: list[np.ndarray] = []
    all_mse_fp4: list[np.ndarray] = []
    all_mse_clip_fp4: list[np.ndarray] = []
    all_mse_clip_nf4: list[np.ndarray] = []
    all_best_pct_fp4: list[np.ndarray] = []
    all_best_pct_nf4: list[np.ndarray] = []
    all_mod: list[np.ndarray] = []
    layer_summaries: list[dict] = []
    mod_types_list = sorted(TARGET_MODULES)
    mod_idx = {m: i for i, m in enumerate(mod_types_list)}

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
            m_nf4 = _nearest_mse(groups, NF4_CODEBOOK)
            m_fp4 = _nearest_mse(groups, FP4_CODEBOOK)

            # clip+FP4
            cfp4 = np.stack(
                [_clip_mse(groups, FP4_CODEBOOK, p) for p in CLIP_PERCENTILES], 0)
            bi_fp4 = np.argmin(cfp4, axis=0)
            m_cfp4 = cfp4[bi_fp4, np.arange(ng)]
            bp_fp4 = np.array([CLIP_PERCENTILES[i] for i in bi_fp4])

            # clip+NF4
            cnf4 = np.stack(
                [_clip_mse(groups, NF4_CODEBOOK, p) for p in CLIP_PERCENTILES], 0)
            bi_nf4 = np.argmin(cnf4, axis=0)
            m_cnf4 = cnf4[bi_nf4, np.arange(ng)]
            bp_nf4 = np.array([CLIP_PERCENTILES[i] for i in bi_nf4])

            all_kappas.append(k)
            all_mse_nf4.append(m_nf4)
            all_mse_fp4.append(m_fp4)
            all_mse_clip_fp4.append(m_cfp4)
            all_mse_clip_nf4.append(m_cnf4)
            all_best_pct_fp4.append(bp_fp4)
            all_best_pct_nf4.append(bp_nf4)
            all_mod.append(np.full(ng, mod_idx[mt], dtype=np.int8))

            elapsed = time.time() - t0
            print(f"  {name}: {ng} grp, "
                  f"k_med={np.median(k):.2f}, "
                  f"nf4={m_nf4.mean():.2e}, "
                  f"clip_nf4={m_cnf4.mean():.2e}  "
                  f"[{elapsed:.0f}s]")

            layer_summaries.append({
                "name": name,
                "mod_type": mt,
                "shape": list(W.shape),
                "n_groups": ng,
                "kurtosis_mean": round(float(k.mean()), 3),
                "kurtosis_median": round(float(np.median(k)), 3),
                "kurtosis_p90": round(float(np.percentile(k, 90)), 3),
                "mse_nf4_mean": float(m_nf4.mean()),
                "mse_fp4_mean": float(m_fp4.mean()),
                "mse_clip_fp4_mean": float(m_cfp4.mean()),
                "mse_clip_nf4_mean": float(m_cnf4.mean()),
            })

            n_done += 1
            if args.layers > 0 and n_done >= args.layers:
                break
        del tensors
        if args.layers > 0 and n_done >= args.layers:
            break

    if not all_kappas:
        sys.exit("No groups processed")

    K = np.concatenate(all_kappas)
    M_nf4 = np.concatenate(all_mse_nf4)
    M_fp4 = np.concatenate(all_mse_fp4)
    M_cfp4 = np.concatenate(all_mse_clip_fp4)
    M_cnf4 = np.concatenate(all_mse_clip_nf4)
    BP_fp4 = np.concatenate(all_best_pct_fp4)
    BP_nf4 = np.concatenate(all_best_pct_nf4)
    MOD = np.concatenate(all_mod)

    total = len(K)
    print(f"\nTotal: {total} groups, {n_done} layers, {time.time()-t0:.0f}s")

    # ── tau sweep: both variants ───────────────────────────────────
    taus = np.arange(-3, 15.1, 0.25)
    sweep_cfp4 = sweep_tau(K, M_nf4, M_cfp4, taus)
    sweep_cnf4 = sweep_tau(K, M_nf4, M_cnf4, taus)
    best_cfp4 = max(sweep_cfp4, key=lambda r: r["improvement_pct"])
    best_cnf4 = max(sweep_cnf4, key=lambda r: r["improvement_pct"])

    print(f"\n{'='*60}")
    print(f"MSE all-NF4:  {M_nf4.mean():.6e}")
    print(f"MSE all-FP4:  {M_fp4.mean():.6e}")
    print(f"\n--- clip+FP4 (original KAQ) ---")
    print(f"BEST tau={best_cfp4['tau']:.2f}:")
    print(f"  MSE:         {best_cfp4['mse_kaq']:.6e}")
    print(f"  Improvement: {best_cfp4['improvement_pct']:.2f}%")
    print(f"  High-k grp:  {best_cfp4['n_high']}/{total} "
          f"({best_cfp4['frac_high']*100:.1f}%)")
    print(f"\n--- clip+NF4 ---")
    print(f"BEST tau={best_cnf4['tau']:.2f}:")
    print(f"  MSE:         {best_cnf4['mse_kaq']:.6e}")
    print(f"  Improvement: {best_cnf4['improvement_pct']:.2f}%")
    print(f"  High-k grp:  {best_cnf4['n_high']}/{total} "
          f"({best_cnf4['frac_high']*100:.1f}%)")
    print(f"{'='*60}")

    # ── per-module-type correlation ────────────────────────────────
    from scipy.stats import spearmanr
    print("\nSpearman(kurtosis, MSE_nf4) per module type:")
    mod_corr = {}
    for mt in mod_types_list:
        mi = mod_idx[mt]
        mask = MOD == mi
        if mask.sum() < 10:
            continue
        rho, p = spearmanr(K[mask], M_nf4[mask])
        print(f"  {mt:12s}: rho={rho:.3f}  p={p:.2e}  n={int(mask.sum())}")
        mod_corr[mt] = {"rho": round(float(rho), 4),
                        "p": float(p), "n": int(mask.sum())}

    # ── kurtosis distribution ──────────────────────────────────────
    k_dist = {
        "mean": round(float(K.mean()), 3),
        "median": round(float(np.median(K)), 3),
        "std": round(float(K.std()), 3),
        "p10": round(float(np.percentile(K, 10)), 3),
        "p25": round(float(np.percentile(K, 25)), 3),
        "p75": round(float(np.percentile(K, 75)), 3),
        "p90": round(float(np.percentile(K, 90)), 3),
        "p99": round(float(np.percentile(K, 99)), 3),
        "frac_gt_1": round(float((K > 1).mean()), 4),
        "frac_gt_3": round(float((K > 3).mean()), 4),
        "frac_gt_5": round(float((K > 5).mean()), 4),
        "frac_gt_10": round(float((K > 10).mean()), 4),
    }

    bp_fp4_dist = {str(p): round(float((BP_fp4 == p).mean()), 4)
                   for p in CLIP_PERCENTILES}
    bp_nf4_dist = {str(p): round(float((BP_nf4 == p).mean()), 4)
                   for p in CLIP_PERCENTILES}

    # ── save ───────────────────────────────────────────────────────
    slug = args.model.replace("/", "_")
    out_path = Path("results") / f"kaq_phase1_{slug}.json"
    out_path.parent.mkdir(exist_ok=True)

    output = {
        "model": args.model,
        "group_size": GROUP_SIZE,
        "n_groups": total,
        "n_layers": n_done,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "kurtosis_distribution": k_dist,
        "mse_all_nf4": float(M_nf4.mean()),
        "mse_all_fp4": float(M_fp4.mean()),
        "clip_fp4": {
            "best_tau": best_cfp4,
            "tau_sweep": sweep_cfp4,
            "best_clip_pct_distribution": bp_fp4_dist,
        },
        "clip_nf4": {
            "best_tau": best_cnf4,
            "tau_sweep": sweep_cnf4,
            "best_clip_pct_distribution": bp_nf4_dist,
        },
        "module_type_correlation": mod_corr,
        "layer_summaries": layer_summaries,
    }

    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
