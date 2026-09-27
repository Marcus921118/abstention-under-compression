#!/usr/bin/env python3
"""KAQ Deep Analysis: diagnose why custom codebooks destroy accuracy.

Two phases:
  Phase A (CPU-feasible): weight-level diagnostics
    - Per-layer MSE: KAQ vs NF4
    - Tail clipping analysis: how many groups have values in uncovered range
    - Module-type breakdown
    - Fix variant MSE: forced endpoints, selective application

  Phase B (GPU required): end-to-end eval of promising fix variants
    - KAQ-fix1: force ±1.0 endpoints on all codebooks
    - KAQ-fix2: KAQ only on high-kurtosis groups (k≥3), NF4 on rest
    - KAQ-fix3: KAQ only on v_proj layers, NF4 on rest

Usage:
    # Phase A only (CPU ok):
    python experiments/kaq_deep_analysis.py --model Qwen/Qwen2.5-1.5B-Instruct \
        --codebook-json experiments/results/kaq_custom_codebook.json --phase A

    # Phase B (needs GPU):
    python experiments/kaq_deep_analysis.py --model Qwen/Qwen2.5-1.5B-Instruct \
        --codebook-json experiments/results/kaq_custom_codebook.json --phase B \
        --n-samples 2000 --seeds 5

    # Both:
    python experiments/kaq_deep_analysis.py --model Qwen/Qwen2.5-1.5B-Instruct \
        --codebook-json experiments/results/kaq_custom_codebook.json --phase AB \
        --n-samples 2000 --seeds 5
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from aq.quant.kaq import (
    CodebookSet, NF4_CODEBOOK, TARGET_SUFFIXES, GROUP_SIZE,
    fake_quantize_tensor, _kurtosis,
)


def parse_args():
    p = argparse.ArgumentParser(description="KAQ deep analysis")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--codebook-json",
                   default="experiments/results/kaq_custom_codebook.json")
    p.add_argument("--phase", default="AB", choices=["A", "B", "AB"])
    p.add_argument("--n-samples", type=int, default=2000)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--out-dir", default="results/kaq_deep_analysis")
    p.add_argument("--dtype", default="float16",
                   choices=["float16", "bfloat16"])
    return p.parse_args()


# ── Codebook variants ──────────────────────────────────────────

def make_codebook_fix1(cbs_orig: CodebookSet) -> CodebookSet:
    """Fix 1: force ±1.0 endpoints on all codebooks."""
    new_cbs = []
    for cb in cbs_orig.codebooks:
        cb_new = cb.clone()
        cb_new[0] = -1.0
        cb_new[-1] = 1.0
        new_cbs.append(cb_new)
    return CodebookSet(
        codebooks=new_cbs,
        kappa_edges=cbs_orig.kappa_edges,
        group_size=cbs_orig.group_size,
        label="kaq-fix1-endpoints",
    )


def make_codebook_fix2(cbs_orig: CodebookSet) -> CodebookSet:
    """Fix 2: KAQ only on high-kurtosis groups (k>=3), NF4 on rest.

    We achieve this by replacing bins 0,1,2 with NF4 codebook.
    """
    nf4_cb = torch.tensor(NF4_CODEBOOK, dtype=torch.float32)
    new_cbs = []
    for i, cb in enumerate(cbs_orig.codebooks):
        if i < len(cbs_orig.codebooks) - 1:
            new_cbs.append(nf4_cb.clone())
        else:
            new_cbs.append(cb.clone())
    return CodebookSet(
        codebooks=new_cbs,
        kappa_edges=cbs_orig.kappa_edges,
        group_size=cbs_orig.group_size,
        label="kaq-fix2-selective",
    )


def make_codebook_fix1and2(cbs_orig: CodebookSet) -> CodebookSet:
    """Fix 1+2 combined: force endpoints AND selective application."""
    nf4_cb = torch.tensor(NF4_CODEBOOK, dtype=torch.float32)
    new_cbs = []
    for i, cb in enumerate(cbs_orig.codebooks):
        if i < len(cbs_orig.codebooks) - 1:
            new_cbs.append(nf4_cb.clone())
        else:
            cb_new = cb.clone()
            cb_new[0] = -1.0
            cb_new[-1] = 1.0
            new_cbs.append(cb_new)
    return CodebookSet(
        codebooks=new_cbs,
        kappa_edges=cbs_orig.kappa_edges,
        group_size=cbs_orig.group_size,
        label="kaq-fix1+2",
    )


# ── Phase A: weight-level diagnostics ─────────────────────────

def phase_a(model_id: str, codebook_json: str, out_dir: Path):
    from transformers import AutoModelForCausalLM

    print("\n" + "=" * 60)
    print("PHASE A: Weight-level diagnostics")
    print("=" * 60)

    cbs_kaq = CodebookSet.from_json(codebook_json)
    cbs_nf4 = CodebookSet.nf4()
    cbs_fix1 = make_codebook_fix1(cbs_kaq)
    cbs_fix2 = make_codebook_fix2(cbs_kaq)
    cbs_fix1and2 = make_codebook_fix1and2(cbs_kaq)

    print(f"\nLoading model weights: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, device_map="cpu",
    )

    results = {
        "model": model_id,
        "layers": {},
        "module_type_summary": {},
        "tail_clipping": {},
        "variant_mse": {},
    }

    layer_data = []
    gs = GROUP_SIZE
    nf4_cb = torch.tensor(NF4_CODEBOOK, dtype=torch.float32)

    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf == "lm_head":
            continue
        if not any(name.endswith(s) for s in TARGET_SUFFIXES):
            continue

        w = mod.weight.data.float()
        flat = w.reshape(-1)
        n_groups = len(flat) // gs
        if n_groups == 0:
            continue
        groups = flat[:n_groups * gs].reshape(n_groups, gs)

        kappa = _kurtosis(groups)
        scales = groups.abs().amax(dim=1).clamp(min=1e-30)
        normed = groups / scales.unsqueeze(1)

        # Per-group tail analysis: fraction of values in [0.92, 1.0] normalized range
        tail_mask = normed.abs() > 0.92
        n_tail_values = tail_mask.sum().item()
        n_total_values = normed.numel()
        tail_frac = n_tail_values / n_total_values

        # How many GROUPS have at least one tail value
        groups_with_tail = (tail_mask.any(dim=1)).sum().item()

        # MSE for each variant
        w_nf4 = fake_quantize_tensor(w, cbs_nf4)
        w_kaq = fake_quantize_tensor(w, cbs_kaq)
        w_fix1 = fake_quantize_tensor(w, cbs_fix1)
        w_fix2 = fake_quantize_tensor(w, cbs_fix2)
        w_fix1and2 = fake_quantize_tensor(w, cbs_fix1and2)

        mse_nf4 = float(((w - w_nf4) ** 2).mean())
        mse_kaq = float(((w - w_kaq) ** 2).mean())
        mse_fix1 = float(((w - w_fix1) ** 2).mean())
        mse_fix2 = float(((w - w_fix2) ** 2).mean())
        mse_fix1and2 = float(((w - w_fix1and2) ** 2).mean())

        # Per-group MSE comparison
        # For each group, compute MSE under NF4 vs KAQ
        nf4_grp_mse = []
        kaq_grp_mse = []
        for gi in range(n_groups):
            g_orig = groups[gi]
            g_nf4 = w_nf4.reshape(-1)[gi*gs:(gi+1)*gs]
            g_kaq = w_kaq.reshape(-1)[gi*gs:(gi+1)*gs]
            nf4_grp_mse.append(float(((g_orig - g_nf4) ** 2).mean()))
            kaq_grp_mse.append(float(((g_orig - g_kaq) ** 2).mean()))

        nf4_grp_mse = np.array(nf4_grp_mse)
        kaq_grp_mse = np.array(kaq_grp_mse)
        kaq_wins = (kaq_grp_mse < nf4_grp_mse).sum()
        kaq_loses = (kaq_grp_mse > nf4_grp_mse).sum()

        # Max absolute error analysis (important for model function)
        max_err_nf4 = float((w - w_nf4).abs().max())
        max_err_kaq = float((w - w_kaq).abs().max())
        max_err_fix1 = float((w - w_fix1).abs().max())

        # Kurtosis distribution for this layer
        kappa_np = kappa.numpy()
        kappa_stats = {
            "mean": float(kappa_np.mean()),
            "std": float(kappa_np.std()),
            "p25": float(np.percentile(kappa_np, 25)),
            "p50": float(np.percentile(kappa_np, 50)),
            "p75": float(np.percentile(kappa_np, 75)),
            "p95": float(np.percentile(kappa_np, 95)),
            "frac_k_ge_3": float((kappa_np >= 3).mean()),
            "frac_k_ge_1": float((kappa_np >= 1).mean()),
        }

        module_type = leaf
        entry = {
            "name": name,
            "module_type": module_type,
            "n_groups": n_groups,
            "n_params": w.numel(),
            "mse_nf4": mse_nf4,
            "mse_kaq": mse_kaq,
            "mse_fix1": mse_fix1,
            "mse_fix2": mse_fix2,
            "mse_fix1and2": mse_fix1and2,
            "mse_ratio_kaq_vs_nf4": mse_kaq / max(mse_nf4, 1e-30),
            "mse_ratio_fix1_vs_nf4": mse_fix1 / max(mse_nf4, 1e-30),
            "mse_ratio_fix2_vs_nf4": mse_fix2 / max(mse_nf4, 1e-30),
            "max_err_nf4": max_err_nf4,
            "max_err_kaq": max_err_kaq,
            "max_err_fix1": max_err_fix1,
            "tail_frac": tail_frac,
            "groups_with_tail": groups_with_tail,
            "groups_with_tail_pct": groups_with_tail / n_groups,
            "kaq_wins_groups": int(kaq_wins),
            "kaq_loses_groups": int(kaq_loses),
            "kappa": kappa_stats,
        }

        results["layers"][name] = entry
        layer_data.append(entry)

        print(f"  {name:50s}  "
              f"NF4={mse_nf4:.2e}  KAQ={mse_kaq:.2e} ({mse_kaq/mse_nf4:.3f}x)  "
              f"fix1={mse_fix1:.2e} ({mse_fix1/mse_nf4:.3f}x)  "
              f"tail={tail_frac:.4f}  maxE_nf4={max_err_nf4:.4f} maxE_kaq={max_err_kaq:.4f}")

    # Module-type aggregation
    by_type = defaultdict(list)
    for d in layer_data:
        by_type[d["module_type"]].append(d)

    print(f"\n{'='*60}")
    print("Module-type summary")
    print(f"{'='*60}")

    for mtype, entries in sorted(by_type.items()):
        mean_ratio = np.mean([e["mse_ratio_kaq_vs_nf4"] for e in entries])
        mean_ratio_fix1 = np.mean([e["mse_ratio_fix1_vs_nf4"] for e in entries])
        mean_ratio_fix2 = np.mean([e["mse_ratio_fix2_vs_nf4"] for e in entries])
        mean_tail = np.mean([e["tail_frac"] for e in entries])
        mean_max_err_nf4 = np.mean([e["max_err_nf4"] for e in entries])
        mean_max_err_kaq = np.mean([e["max_err_kaq"] for e in entries])
        mean_k_ge3 = np.mean([e["kappa"]["frac_k_ge_3"] for e in entries])
        total_groups = sum(e["n_groups"] for e in entries)
        total_params = sum(e["n_params"] for e in entries)

        results["module_type_summary"][mtype] = {
            "n_layers": len(entries),
            "total_groups": total_groups,
            "total_params": total_params,
            "mean_mse_ratio_kaq_nf4": round(mean_ratio, 4),
            "mean_mse_ratio_fix1_nf4": round(mean_ratio_fix1, 4),
            "mean_mse_ratio_fix2_nf4": round(mean_ratio_fix2, 4),
            "mean_tail_frac": round(mean_tail, 6),
            "mean_max_err_nf4": round(mean_max_err_nf4, 4),
            "mean_max_err_kaq": round(mean_max_err_kaq, 4),
            "mean_frac_k_ge_3": round(mean_k_ge3, 4),
        }

        print(f"  {mtype:12s}: {len(entries):3d} layers, "
              f"KAQ/NF4={mean_ratio:.3f}x, "
              f"fix1/NF4={mean_ratio_fix1:.3f}x, "
              f"fix2/NF4={mean_ratio_fix2:.3f}x, "
              f"tail={mean_tail:.4f}, "
              f"k≥3={mean_k_ge3:.3f}")

    # Global tail analysis
    all_tail = [e["tail_frac"] for e in layer_data]
    all_tail_groups = [e["groups_with_tail_pct"] for e in layer_data]
    total_groups = sum(e["n_groups"] for e in layer_data)

    results["tail_clipping"] = {
        "mean_tail_value_frac": round(np.mean(all_tail), 6),
        "mean_groups_with_tail_pct": round(np.mean(all_tail_groups), 4),
        "total_groups": total_groups,
    }

    # Variant MSE summary
    for variant_name, mse_key in [
        ("nf4", "mse_nf4"), ("kaq", "mse_kaq"),
        ("fix1_endpoints", "mse_fix1"),
        ("fix2_selective", "mse_fix2"),
        ("fix1+2_combined", "mse_fix1and2"),
    ]:
        mses = [e[mse_key] for e in layer_data]
        results["variant_mse"][variant_name] = {
            "mean": float(np.mean(mses)),
            "total": float(np.sum(mses)),
        }

    print(f"\n{'='*60}")
    print("Variant MSE summary (mean across layers)")
    print(f"{'='*60}")
    for vname, vdata in results["variant_mse"].items():
        ratio = vdata["mean"] / results["variant_mse"]["nf4"]["mean"]
        print(f"  {vname:20s}: {vdata['mean']:.6e}  ({ratio:.4f}x NF4)")

    print(f"\nTail value fraction: {np.mean(all_tail):.4f}")
    print(f"Groups with tail values: {np.mean(all_tail_groups):.4f}")

    # Max error comparison
    print(f"\n{'='*60}")
    print("Max absolute error (critical for model function)")
    print(f"{'='*60}")
    nf4_max_errs = [e["max_err_nf4"] for e in layer_data]
    kaq_max_errs = [e["max_err_kaq"] for e in layer_data]
    fix1_max_errs = [e["max_err_fix1"] for e in layer_data]
    print(f"  NF4  max abs err: mean={np.mean(nf4_max_errs):.4f}, "
          f"p95={np.percentile(nf4_max_errs, 95):.4f}, max={np.max(nf4_max_errs):.4f}")
    print(f"  KAQ  max abs err: mean={np.mean(kaq_max_errs):.4f}, "
          f"p95={np.percentile(kaq_max_errs, 95):.4f}, max={np.max(kaq_max_errs):.4f}")
    print(f"  Fix1 max abs err: mean={np.mean(fix1_max_errs):.4f}, "
          f"p95={np.percentile(fix1_max_errs, 95):.4f}, max={np.max(fix1_max_errs):.4f}")

    # Cleanup
    del model
    gc.collect()

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "phase_a.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved: {out_dir / 'phase_a.json'}")
    return results


# ── Phase B: end-to-end eval of fix variants ───────────────────

def apply_fix3_vproj_only(model, cbs_kaq: CodebookSet, cbs_nf4: CodebookSet):
    """Fix 3: KAQ on v_proj only, NF4 on all other layers."""
    mse_log = {}
    n = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf == "lm_head":
            continue
        if not any(name.endswith(s) for s in TARGET_SUFFIXES):
            continue

        w_orig = mod.weight.data
        w_cpu = w_orig.cpu()
        if name.endswith("v_proj"):
            w_quant = fake_quantize_tensor(w_cpu, cbs_kaq)
        else:
            w_quant = fake_quantize_tensor(w_cpu, cbs_nf4)
        mse = float(((w_cpu.float() - w_quant.float()) ** 2).mean())
        mod.weight.data = w_quant.to(w_orig.device)
        mse_log[name] = mse
        n += 1

    total_mse = sum(mse_log.values()) / max(len(mse_log), 1)
    print(f"  Fix3 (v_proj KAQ, rest NF4) applied: {n} layers, mean MSE={total_mse:.6e}")
    return mse_log


def phase_b(model_id: str, codebook_json: str, out_dir: Path,
            n_samples: int, seeds: int, batch_size: int, max_new_tokens: int,
            dtype_name: str = "float16"):
    import importlib.util
    _kaq_p2 = _ROOT / "experiments" / "kaq_phase2.py"
    spec = importlib.util.spec_from_file_location("kaq_phase2", _kaq_p2)
    kaq_phase2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kaq_phase2)
    load_model_fp16 = kaq_phase2.load_model_fp16
    release_model = kaq_phase2.release_model
    load_samples = kaq_phase2.load_samples
    run_generation = kaq_phase2.run_generation
    compute_metrics = kaq_phase2.compute_metrics

    print("\n" + "=" * 60)
    print("PHASE B: End-to-end eval of fix variants")
    print("=" * 60)

    cbs_kaq = CodebookSet.from_json(codebook_json)
    cbs_nf4 = CodebookSet.nf4()
    cbs_fix1 = make_codebook_fix1(cbs_kaq)
    cbs_fix2 = make_codebook_fix2(cbs_kaq)
    cbs_fix1and2 = make_codebook_fix1and2(cbs_kaq)

    conditions = {
        "fp16": ("FP16 (no quant)", None),
        "nf4": ("NF4 baseline", cbs_nf4),
        "kaq-fix1": ("KAQ + forced endpoints", cbs_fix1),
        "kaq-fix2": ("KAQ selective (k>=3 only)", cbs_fix2),
        "kaq-fix3": ("KAQ v_proj only", None),  # special handling
        "kaq-fix1+2": ("KAQ endpoints + selective", cbs_fix1and2),
    }

    all_metrics = {}
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in range(seeds):
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        print(f"{'='*60}")

        samples = load_samples(n_samples, seed=seed)
        print(f"  Loaded {len(samples)} samples")

        for cond_name, (desc, cbs) in conditions.items():
            tag = f"{cond_name}_seed{seed}"
            raw_path = out_dir / f"{tag}.jsonl"

            if raw_path.exists() and raw_path.stat().st_size > 0:
                print(f"\n  [{cond_name}] cached: {raw_path}")
                records = [json.loads(l)
                           for l in raw_path.read_text().splitlines() if l.strip()]
            else:
                print(f"\n  [{cond_name}] {desc}")
                model, tokenizer = load_model_fp16(model_id, dtype_name)

                if cond_name == "fp16":
                    pass
                elif cond_name == "kaq-fix3":
                    apply_fix3_vproj_only(model, cbs_kaq, cbs_nf4)
                elif cbs is not None:
                    from aq.quant.kaq import apply_fake_quant
                    apply_fake_quant(model, cbs)

                print(f"  Running generation ({len(samples)} samples)...")
                t0 = time.time()
                records = run_generation(
                    model, tokenizer, samples,
                    batch_size=batch_size,
                    max_new_tokens=max_new_tokens,
                )
                elapsed = time.time() - t0
                print(f"  Done in {elapsed:.1f}s")

                with open(raw_path, "w") as f:
                    for r in records:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")

                release_model(model)

            metrics = compute_metrics(records)
            all_metrics[tag] = metrics
            print(f"  [{cond_name}] acc={metrics['accuracy_on_answerable']:.3f}  "
                  f"ECE={metrics['ece_10bin']:.4f}  "
                  f"overconf={metrics['overconfidence_rate_tau0.8']:.3f}")

    # Aggregate
    print(f"\n\n{'='*60}")
    print(f"AGGREGATE ({seeds} seeds)")
    print(f"{'='*60}\n")

    summary = {}
    for cond_name in conditions:
        seed_metrics = [all_metrics[f"{cond_name}_seed{s}"]
                        for s in range(seeds)]
        agg = {}
        for key in seed_metrics[0]:
            vals = [m[key] for m in seed_metrics if m[key] is not None]
            if vals and isinstance(vals[0], (int, float)):
                agg[key] = {
                    "mean": round(float(np.mean(vals)), 4),
                    "std": round(float(np.std(vals, ddof=1)), 4) if len(vals) > 1 else 0,
                }
        summary[cond_name] = agg
        print(f"  {cond_name:15s}: "
              f"acc={agg['accuracy_on_answerable']['mean']:.3f} "
              f"(+/-{agg['accuracy_on_answerable']['std']:.3f})  "
              f"ECE={agg['ece_10bin']['mean']:.4f}  "
              f"overconf={agg['overconfidence_rate_tau0.8']['mean']:.3f}")

    # Deltas vs NF4
    nf4_acc = summary["nf4"]["accuracy_on_answerable"]["mean"]
    nf4_oc = summary["nf4"]["overconfidence_rate_tau0.8"]["mean"]
    print(f"\n  Deltas vs NF4:")
    for cond_name in conditions:
        if cond_name == "nf4":
            continue
        d_acc = summary[cond_name]["accuracy_on_answerable"]["mean"] - nf4_acc
        d_oc = summary[cond_name]["overconfidence_rate_tau0.8"]["mean"] - nf4_oc
        print(f"    {cond_name:15s}: acc={d_acc:+.4f}  overconf={d_oc:+.4f}")

    out = {
        "model": model_id,
        "n_samples": n_samples,
        "n_seeds": seeds,
        "conditions": list(conditions.keys()),
        "per_seed": all_metrics,
        "aggregate": summary,
    }
    (out_dir / "phase_b.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nSaved: {out_dir / 'phase_b.json'}")
    return summary


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)

    if "A" in args.phase:
        phase_a(args.model, args.codebook_json, out_dir)

    if "B" in args.phase:
        phase_b(args.model, args.codebook_json, out_dir,
                args.n_samples, args.seeds, args.batch_size,
                args.max_new_tokens, dtype_name=args.dtype)


if __name__ == "__main__":
    main()
