#!/usr/bin/env python3
"""KAQ Phase 2: end-to-end overconfidence evaluation.

Compares FP16 vs NF4-fake-quant vs KAQ-fake-quant on MCQA (MMLU).
Measures accuracy, calibration, and overconfidence.

Server deployment:
    python experiments/kaq_phase2.py \\
        --model Qwen/Qwen2.5-1.5B-Instruct \\
        --codebook-json experiments/results/kaq_custom_codebook.json \\
        --n-samples 2000 \\
        --seeds 5

Conditions run sequentially (one model in VRAM at a time):
  1. FP16 (no quant)
  2. NF4 fake quant (absmax + NF4 codebook, gs=128)
  3. KAQ fake quant (absmax + kappa-adaptive codebooks, gs=128)
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

# Ensure project root on path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))


def parse_args():
    p = argparse.ArgumentParser(description="KAQ Phase 2 evaluation")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--codebook-json",
                   default="experiments/results/kaq_custom_codebook.json")
    p.add_argument("--n-samples", type=int, default=2000)
    p.add_argument("--seeds", type=int, default=5,
                   help="number of data seeds (each draws different samples)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--out-dir", default="results/kaq_phase2")
    p.add_argument("--conditions", nargs="+",
                   default=["fp16", "nf4", "kaq"],
                   choices=["fp16", "nf4", "kaq"])
    return p.parse_args()


def load_model_fp16(model_id: str, dtype_name: str = "float16"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_name]
    print(f"  Loading {model_id} ({dtype_name})...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, device_map="auto",
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Loaded in {time.time()-t0:.1f}s, "
          f"device={next(model.parameters()).device}")
    return model, tokenizer


def release_model(model):
    import torch
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_samples(n_samples: int, seed: int):
    """Load MCQA MMLU samples using the project's data loader."""
    from aq.config import DataConfig
    from aq.data.mcqa_mmlu import load

    cfg = DataConfig(name="mcqa_mmlu", split="test", max_samples=n_samples,
                     seed=seed)
    return load(cfg)


def run_generation(model, tokenizer, samples, *, batch_size=8,
                   max_new_tokens=64, verbose=True):
    """Run generation on samples, return list of result dicts."""
    import torch
    from aq.confidence.token_prob import sequence_confidence_from_logprobs
    from aq.confidence.verbalized import parse_verbalized_confidence
    from aq.eval.mcqa import mcqa_outcome
    from aq.eval.pipeline import _build_messages, SYSTEM_PROMPT

    device = next(model.parameters()).device
    tokenizer.padding_side = "left"
    results = []

    for start in range(0, len(samples), batch_size):
        end = min(start + batch_size, len(samples))
        batch = samples[start:end]
        bs = len(batch)

        all_ids = []
        for s in batch:
            msgs = _build_messages(s.question)
            enc = tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True,
                return_tensors="pt", return_dict=True,
            )
            all_ids.append(enc["input_ids"].squeeze(0))

        max_len = max(t.shape[0] for t in all_ids)
        pad_id = tokenizer.pad_token_id
        padded = torch.full((bs, max_len), pad_id, dtype=all_ids[0].dtype,
                            device=device)
        attn = torch.zeros((bs, max_len), dtype=torch.long, device=device)
        input_lens = []
        for i, ids in enumerate(all_ids):
            seq_len = ids.shape[0]
            padded[i, max_len - seq_len:] = ids.to(device)
            attn[i, max_len - seq_len:] = 1
            input_lens.append(seq_len)

        with torch.no_grad():
            out = model.generate(
                input_ids=padded, attention_mask=attn,
                max_new_tokens=max_new_tokens, do_sample=False,
                return_dict_in_generate=True, output_scores=True,
                pad_token_id=pad_id,
            )

        seqs = out.sequences.cpu()
        trans = model.compute_transition_scores(
            out.sequences, out.scores, normalize_logits=True
        ).cpu()
        del out
        torch.cuda.empty_cache()

        for i in range(bs):
            gen_ids = seqs[i, max_len:]
            eos_mask = gen_ids != pad_id
            if tokenizer.eos_token_id is not None:
                eos_mask = eos_mask & (gen_ids != tokenizer.eos_token_id)
            gen_ids = gen_ids[eos_mask]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            n_gen = gen_ids.shape[0]
            token_lps = [float(x) for x in trans[i].tolist()[:n_gen]
                         if x == x and x != float("-inf")]

            s = batch[i]
            tok_conf = sequence_confidence_from_logprobs(token_lps)
            verb_conf = parse_verbalized_confidence(text)
            mc = mcqa_outcome(text,
                              s.meta.get("gold_letter", ""),
                              s.meta.get("abstain_letter", "D"))

            results.append({
                "sample_id": s.id,
                "answerable": s.answerable,
                "category": s.category,
                "output_text": text,
                "choice": mc["choice"],
                "abstained": mc["abstained"],
                "correct": mc["correct"],
                "token_prob_confidence": tok_conf,
                "verbalized_confidence": verb_conf,
                "mean_logprob": (sum(token_lps) / len(token_lps))
                                if token_lps else None,
            })

        if verbose and end % 100 < batch_size:
            print(f"    [{end}/{len(samples)}]")

    return results


def compute_metrics(records: list[dict]) -> dict:
    """Compute accuracy, calibration, and overconfidence metrics."""
    import numpy as np

    answerable = [r for r in records if r["answerable"]]
    unanswerable = [r for r in records if not r["answerable"]]

    # accuracy on answerable (non-abstained)
    ans_responded = [r for r in answerable if not r["abstained"]]
    n_correct = sum(1 for r in ans_responded if r["correct"])
    acc_on_ans = n_correct / max(len(ans_responded), 1)

    # abstention rates
    abs_rate_ans = sum(1 for r in answerable if r["abstained"]) / max(len(answerable), 1)
    abs_rate_unans = sum(1 for r in unanswerable if r["abstained"]) / max(len(unanswerable), 1)

    # overconfidence: P(conf > tau | incorrect)
    tau = 0.8
    incorrect = [r for r in ans_responded
                 if r["correct"] is False and r["token_prob_confidence"] is not None]
    overconf_rate = (sum(1 for r in incorrect if r["token_prob_confidence"] > tau)
                     / max(len(incorrect), 1))

    # mean confidence on correct vs incorrect
    correct_confs = [r["token_prob_confidence"] for r in ans_responded
                     if r["correct"] and r["token_prob_confidence"] is not None]
    incorrect_confs = [r["token_prob_confidence"] for r in incorrect]

    # ECE (10 bins)
    all_responded = [r for r in ans_responded
                     if r["token_prob_confidence"] is not None and r["correct"] is not None]
    ece = 0.0
    if all_responded:
        confs = np.array([r["token_prob_confidence"] for r in all_responded])
        accs = np.array([float(r["correct"]) for r in all_responded])
        bins_edges = np.linspace(0, 1, 11)
        for lo, hi in zip(bins_edges[:-1], bins_edges[1:]):
            mask = (confs >= lo) & (confs < hi)
            if mask.sum() > 0:
                bin_conf = confs[mask].mean()
                bin_acc = accs[mask].mean()
                ece += mask.sum() / len(confs) * abs(bin_acc - bin_conf)

    return {
        "n_total": len(records),
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "accuracy_on_answerable": round(acc_on_ans, 4),
        "abstention_rate_answerable": round(abs_rate_ans, 4),
        "abstention_rate_unanswerable": round(abs_rate_unans, 4),
        "overconfidence_rate_tau0.8": round(overconf_rate, 4),
        "mean_conf_correct": round(float(np.mean(correct_confs)), 4) if correct_confs else None,
        "mean_conf_incorrect": round(float(np.mean(incorrect_confs)), 4) if incorrect_confs else None,
        "ece_10bin": round(float(ece), 4),
        "n_incorrect": len(incorrect),
    }


def run_condition(condition: str, model_id: str, samples, codebook_json: str,
                  batch_size: int, max_new_tokens: int):
    """Load model, optionally apply fake quant, run generation."""
    from aq.quant.kaq import CodebookSet, apply_fake_quant

    model, tokenizer = load_model_fp16(model_id)

    if condition == "nf4":
        print(f"  Applying NF4 fake quantization...")
        cbs = CodebookSet.nf4()
        apply_fake_quant(model, cbs)
    elif condition == "kaq":
        print(f"  Applying KAQ fake quantization...")
        cbs = CodebookSet.from_json(codebook_json)
        apply_fake_quant(model, cbs)

    print(f"  Running generation ({len(samples)} samples)...")
    t0 = time.time()
    results = run_generation(model, tokenizer, samples,
                             batch_size=batch_size,
                             max_new_tokens=max_new_tokens)
    elapsed = time.time() - t0
    print(f"  Generation done in {elapsed:.1f}s")

    release_model(model)
    return results


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"KAQ Phase 2 Evaluation")
    print(f"  Model: {args.model}")
    print(f"  Conditions: {args.conditions}")
    print(f"  Samples: {args.n_samples} x {args.seeds} seeds\n")

    all_metrics = {}

    for seed in range(args.seeds):
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        print(f"{'='*60}")

        samples = load_samples(args.n_samples, seed=seed)
        print(f"  Loaded {len(samples)} samples")

        for cond in args.conditions:
            tag = f"{cond}_seed{seed}"
            raw_path = out_dir / f"{tag}.jsonl"

            if raw_path.exists() and raw_path.stat().st_size > 0:
                print(f"\n  [{cond}] cached: {raw_path}")
                records = [json.loads(l) for l in raw_path.read_text().splitlines() if l.strip()]
            else:
                print(f"\n  [{cond}] running...")
                records = run_condition(
                    cond, args.model, samples, args.codebook_json,
                    args.batch_size, args.max_new_tokens,
                )
                with open(raw_path, "w") as f:
                    for r in records:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                print(f"  Saved {len(records)} records -> {raw_path}")

            metrics = compute_metrics(records)
            all_metrics[tag] = metrics
            print(f"  [{cond}] acc={metrics['accuracy_on_answerable']:.3f}  "
                  f"ECE={metrics['ece_10bin']:.4f}  "
                  f"overconf={metrics['overconfidence_rate_tau0.8']:.3f}  "
                  f"abs_ans={metrics['abstention_rate_answerable']:.3f}")

    # ── Aggregate across seeds ────────────────────────────────────
    import numpy as np

    print(f"\n\n{'='*60}")
    print(f"AGGREGATE ({args.seeds} seeds)")
    print(f"{'='*60}\n")

    summary = {}
    for cond in args.conditions:
        seed_metrics = [all_metrics[f"{cond}_seed{s}"] for s in range(args.seeds)]
        agg = {}
        for key in seed_metrics[0]:
            vals = [m[key] for m in seed_metrics if m[key] is not None]
            if vals and isinstance(vals[0], (int, float)):
                agg[key] = {
                    "mean": round(float(np.mean(vals)), 4),
                    "std": round(float(np.std(vals, ddof=1)), 4) if len(vals) > 1 else 0,
                    "values": [round(v, 4) for v in vals],
                }
        summary[cond] = agg
        print(f"  {cond:>5s}: acc={agg['accuracy_on_answerable']['mean']:.3f} "
              f"(+/-{agg['accuracy_on_answerable']['std']:.3f})  "
              f"ECE={agg['ece_10bin']['mean']:.4f}  "
              f"overconf={agg['overconfidence_rate_tau0.8']['mean']:.3f}")

    # Deltas
    if "fp16" in summary and "nf4" in summary:
        d_acc = (summary["nf4"]["accuracy_on_answerable"]["mean"]
                 - summary["fp16"]["accuracy_on_answerable"]["mean"])
        d_oc = (summary["nf4"]["overconfidence_rate_tau0.8"]["mean"]
                - summary["fp16"]["overconfidence_rate_tau0.8"]["mean"])
        print(f"\n  NF4 vs FP16:  acc delta={d_acc:+.4f}  overconf delta={d_oc:+.4f}")
    if "fp16" in summary and "kaq" in summary:
        d_acc = (summary["kaq"]["accuracy_on_answerable"]["mean"]
                 - summary["fp16"]["accuracy_on_answerable"]["mean"])
        d_oc = (summary["kaq"]["overconfidence_rate_tau0.8"]["mean"]
                - summary["fp16"]["overconfidence_rate_tau0.8"]["mean"])
        print(f"  KAQ vs FP16:  acc delta={d_acc:+.4f}  overconf delta={d_oc:+.4f}")
    if "nf4" in summary and "kaq" in summary:
        d_acc = (summary["kaq"]["accuracy_on_answerable"]["mean"]
                 - summary["nf4"]["accuracy_on_answerable"]["mean"])
        d_oc = (summary["kaq"]["overconfidence_rate_tau0.8"]["mean"]
                - summary["nf4"]["overconfidence_rate_tau0.8"]["mean"])
        print(f"  KAQ vs NF4:   acc delta={d_acc:+.4f}  overconf delta={d_oc:+.4f}")

    # Save
    out = {
        "model": args.model,
        "n_samples": args.n_samples,
        "n_seeds": args.seeds,
        "conditions": args.conditions,
        "per_seed": all_metrics,
        "aggregate": summary,
    }
    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
