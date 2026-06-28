"""CLI entry: selftest / slice / run-baseline / run-quant / run-matrix / analyze.

- selftest : exercise the metric+protocol stack on synthetic data (no model).
- slice    : Gate G2 vertical slice -- one model x {fp16, bnb-nf4 4bit} x one
             dataset, scored under one confidence signal, with a degradation
             curve written to results/figures/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .config import load_run_config
from .eval import metrics as M
from .eval import protocol as P

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def _selftest(_args: argparse.Namespace) -> int:
    rng = np.random.default_rng(0)
    n = 200
    answerable = rng.random(n) > 0.5
    abst_fp16 = np.where(~answerable, rng.random(n) < 0.85, rng.random(n) < 0.05)
    correct = rng.random(n) < 0.75
    conf = np.clip(rng.normal(0.7, 0.15, n), 0, 1)
    fp16 = M.compute_run_metrics(answerable, abst_fp16, correct, conf)
    abst_q = np.where(~answerable, rng.random(n) < 0.30, rng.random(n) < 0.02)
    conf_q = np.clip(rng.normal(0.85, 0.10, n), 0, 1)
    q3 = M.compute_run_metrics(answerable, abst_q, correct, conf_q)
    sdi = P.selective_damage_index(fp16, q3)
    dec = P.decoupling_index(fp16, q3)
    print("[selftest] FP16 abstention_f1 =", round(fp16["abstention_f1"], 3))
    print("[selftest] 3bit abstention_f1 =", round(q3["abstention_f1"], 3))
    print("[selftest] selective_damage_index =", sdi["selective_damage_index"])
    print("[selftest] decoupling_index =", dec["decoupling_index"])
    print("[selftest] OK")
    return 0


def _json_default(o):
    """Coerce numpy scalars/arrays to JSON-native types."""
    if hasattr(o, "item"):
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "  nan"
    if isinstance(v, float):
        return f"{v:6.3f}"
    return str(v)


def _report_sweep(model: str, dataset: str, signal: str, n, kind: str,
                  results: dict[int, dict]) -> int:
    """Shared reporting: per-bit table, protocol indices, degradation curve, JSON."""
    from .analysis.plots import degradation_curve  # noqa: PLC0415

    bits_sorted = sorted(results.keys(), reverse=True)
    metrics_to_show = [
        "abstention_f1", "refuse_when_unanswerable_rate", "accuracy_on_answered",
        "over_confident_wrong_rate", "auroc_conf_vs_correct", "ece", "brier",
    ]
    print(f"\n=== {kind}: {model} on {dataset} (signal={signal}, n={n}) ===")
    cols = "".join(f" | {f'{b}-bit':>9s}" for b in bits_sorted)
    header = f"{'metric':32s}{cols}"
    print(header)
    print("-" * len(header))
    for k in metrics_to_show:
        row = "".join(f" | {_fmt(results[b].get(k)):>9s}" for b in bits_sorted)
        print(f"{k:32s}{row}")

    sdi = P.selective_damage_index(results[bits_sorted[0]], results[bits_sorted[-1]])
    dec = P.decoupling_index(results[bits_sorted[0]], results[bits_sorted[-1]])
    ocw = [results[b].get("over_confident_wrong_rate", float("nan")) for b in bits_sorted]
    cliff = P.find_overconfidence_cliff(bits_sorted, ocw)
    print(f"\n--- protocol indices ({bits_sorted[0]}-bit -> {bits_sorted[-1]}-bit) ---")
    print(f"abstention_f1_drop       = {_fmt(sdi['abstention_f1_drop'])}")
    print(f"accuracy_drop            = {_fmt(sdi['accuracy_drop'])}")
    print(f"selective_damage_index   = {sdi['selective_damage_index']}")
    print(f"decoupling_index         = {dec['decoupling_index']}")
    print(f"overconfidence_cliff_bits= {cliff.cliff_bits} (max_jump={_fmt(cliff.max_jump)})")

    fig_dir = Path("results") / "figures"
    series = {m: [results[b].get(m, float("nan")) for b in bits_sorted]
              for m in ("abstention_f1", "accuracy_on_answered", "over_confident_wrong_rate")}
    out = degradation_curve(
        bits_sorted, series, fig_dir / f"{kind}_{model}_{dataset}.png",
        title=f"{model} on {dataset}: {kind} bit-width sweep",
    )
    print(f"\n[plot] degradation curve -> {out}")

    summary = {"model": model, "dataset": dataset, "signal": signal, "kind": kind,
               "n": n, "bits": bits_sorted,
               "results": {str(b): results[b] for b in bits_sorted},
               "selective_damage": sdi, "decoupling": dec, "cliff_bits": cliff.cliff_bits}
    sjson = Path("results") / f"{kind}_{model}_{dataset}.json"
    sjson.parent.mkdir(parents=True, exist_ok=True)
    sjson.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    print(f"[summary] -> {sjson}")
    return 0


def _slice(args: argparse.Namespace) -> int:
    from .eval.pipeline import run_inference  # noqa: PLC0415
    from .eval.scoring import score_run  # noqa: PLC0415

    bits_list = [int(b) for b in str(args.bits).split(",")]
    data_spec = {"name": args.dataset, "split": args.split, "max_samples": args.max_samples}
    # bnb backend: 16 -> fp16; 8 -> int8; 4 -> NF4. (bnb supports {4,8}.)
    results: dict[int, dict] = {}
    for bits in bits_list:
        quant_name = "fp16" if bits == 16 else "bnb_nf4"
        cfg = load_run_config(
            base=CONFIGS / "base.yaml",
            model=CONFIGS / "models" / f"{args.model}.yaml",
            quant=CONFIGS / "quant" / f"{quant_name}.yaml",
            data=data_spec,
            overrides={"quant": {"bits": bits}, "max_new_tokens": args.max_new_tokens,
                       "seed": args.seed},
        )
        results[bits] = score_run(run_inference(cfg), signal=args.signal)
    return _report_sweep(args.model, args.dataset, args.signal, args.max_samples, "bnb", results)


def _gguf(args: argparse.Namespace) -> int:
    """GGUF/llama.cpp bit-width sweep. --files is 'bits:filename,bits:filename,...'."""
    from .eval.pipeline import run_inference  # noqa: PLC0415
    from .eval.scoring import score_run  # noqa: PLC0415

    pairs = [p.split(":", 1) for p in args.files.split(",")]
    data_spec = {"name": args.dataset, "split": args.split, "max_samples": args.max_samples}
    results: dict[int, dict] = {}
    for bits_s, fname in pairs:
        bits = int(bits_s)
        cfg = load_run_config(
            base=CONFIGS / "base.yaml",
            model=CONFIGS / "models" / f"{args.model}.yaml",
            quant=CONFIGS / "quant" / "gguf.yaml",
            data=data_spec,
            overrides={"quant": {"bits": bits, "gguf_repo": args.gguf_repo, "gguf_file": fname},
                       "max_new_tokens": args.max_new_tokens, "seed": args.seed},
        )
        results[bits] = score_run(run_inference(cfg), signal=args.signal)
    return _report_sweep(args.model, args.dataset, args.signal, args.max_samples, "gguf", results)


def _analyze(args: argparse.Namespace) -> int:
    from .analysis.report import analyze  # noqa: PLC0415

    report = analyze(results_dir=args.results_dir)
    print(report)
    print(f"\n[report] -> {Path(args.results_dir) / 'analysis_report.md'}")
    return 0


def _significance(args: argparse.Namespace) -> int:
    from .analysis.report import significance_report  # noqa: PLC0415

    text = significance_report(args.results_dir, signal=args.signal, n_boot=args.n_boot)
    print(text)
    print(f"\n[report] -> {Path(args.results_dir) / 'significance_report.md'}")
    return 0


def _signals(args: argparse.Namespace) -> int:
    from .analysis.report import signals_report  # noqa: PLC0415

    text = signals_report(args.results_dir)
    print(text)
    print(f"\n[report] -> {Path(args.results_dir) / 'signals_report.md'}")
    return 0


def _controls(args: argparse.Namespace) -> int:
    from .analysis.report import controls_report  # noqa: PLC0415

    text = controls_report(args.results_dir)
    print(text)
    print(f"\n[report] -> {Path(args.results_dir) / 'controls_report.md'}")
    return 0


def _not_ready(name: str):
    def _run(_args: argparse.Namespace) -> int:
        print(f"[{name}] arrives in a later stage (Stage 3).", file=sys.stderr)
        return 2
    return _run


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aq", description="compression-aware abstention eval")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest", help="run metric layer on synthetic data").set_defaults(
        func=_selftest
    )
    sp = sub.add_parser("slice", help="bnb bit-width sweep (fp16/int8/nf4-4bit)")
    sp.add_argument("--model", default="qwen25_15b", help="model config name in configs/models/")
    sp.add_argument("--dataset", default="bootstrap")
    sp.add_argument("--bits", default="16,8,4", help="comma-separated bit-widths (16=fp16)")
    sp.add_argument("--split", default="test")
    sp.add_argument("--signal", default="token_prob", choices=["token_prob", "verbalized"])
    sp.add_argument("--max-samples", type=int, default=None)
    sp.add_argument("--max-new-tokens", type=int, default=128)
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(func=_slice)

    gp = sub.add_parser("gguf", help="GGUF/llama.cpp bit-width sweep")
    gp.add_argument("--model", default="qwen25_15b")
    gp.add_argument("--dataset", default="bootstrap")
    gp.add_argument("--gguf-repo", default="Qwen/Qwen2.5-1.5B-Instruct-GGUF")
    gp.add_argument(
        "--files",
        default=("8:qwen2.5-1.5b-instruct-q8_0.gguf,"
                 "4:qwen2.5-1.5b-instruct-q4_k_m.gguf,"
                 "3:qwen2.5-1.5b-instruct-q3_k_m.gguf,"
                 "2:qwen2.5-1.5b-instruct-q2_k.gguf"),
        help="comma-separated 'bits:filename' pairs",
    )
    gp.add_argument("--split", default="test")
    gp.add_argument("--signal", default="token_prob", choices=["token_prob", "verbalized"])
    gp.add_argument("--max-samples", type=int, default=None)
    gp.add_argument("--max-new-tokens", type=int, default=128)
    gp.add_argument("--seed", type=int, default=0)
    gp.set_defaults(func=_gguf)

    ap = sub.add_parser("analyze", help="aggregate results/*.json -> tables + plots + report")
    ap.add_argument("--results-dir", default="results")
    ap.set_defaults(func=_analyze)

    sg = sub.add_parser("significance", help="paired-bootstrap significance hi-bit vs lo-bit")
    sg.add_argument("--results-dir", default="results")
    sg.add_argument("--signal", default="token_prob", choices=["token_prob", "verbalized"])
    sg.add_argument("--n-boot", type=int, default=2000)
    sg.set_defaults(func=_significance)

    sgn = sub.add_parser("signals", help="compare token-prob vs verbalized confidence")
    sgn.add_argument("--results-dir", default="results")
    sgn.set_defaults(func=_signals)

    cn = sub.add_parser("controls", help="confound controls: format/degeneration/length")
    cn.add_argument("--results-dir", default="results")
    cn.set_defaults(func=_controls)

    sub.add_parser("run-matrix", help="(use scripts/run_matrix.sh)").set_defaults(
        func=_not_ready("run-matrix")
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
