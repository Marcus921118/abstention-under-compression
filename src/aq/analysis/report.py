"""Aggregate sweep summaries (results/{bnb,gguf}_*.json) into a cross-method
comparison: per-dataset tables, over-confidence-cliff summary, overlay plots,
and a markdown report.

Pure-ish: reads JSON summaries produced by `aq slice` / `aq gguf`, writes
figures + a markdown report. No model/torch dependency.
"""

from __future__ import annotations

import json
from pathlib import Path

from .plots import overlay_vs_bits

# metrics carried per bit in each summary's results[bits]
HEADLINE = [
    "abstention_f1", "refuse_when_unanswerable_rate", "accuracy_on_answered",
    "over_confident_wrong_rate", "auroc_conf_vs_correct", "ece", "brier",
]


def load_summaries(results_dir: str | Path) -> list[dict]:
    """Load all bnb_*/gguf_* sweep summary JSONs (skip bootstrap toy runs)."""
    out = []
    for p in sorted(Path(results_dir).glob("*.json")):
        if p.name.startswith(("bnb_", "gguf_")) and "bootstrap" not in p.name:
            try:
                out.append(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                pass
    return out


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "  nan"
    return f"{v:6.3f}" if isinstance(v, (int, float)) else str(v)


def _series(summary: dict, metric: str) -> dict[int, float]:
    res = summary["results"]
    return {int(b): res[str(b)].get(metric) for b in summary["bits"]}


def analyze(results_dir: str | Path = "results", fig_dir: str | Path | None = None) -> str:
    """Build comparison tables + overlay plots + markdown report. Returns the report text."""
    results_dir = Path(results_dir)
    fig_dir = Path(fig_dir) if fig_dir else results_dir / "figures"
    summaries = load_summaries(results_dir)

    # group by (dataset, model) -> {method: summary} so multiple models on the
    # same dataset don't overwrite each other (per-model tables + plots).
    by_dm: dict[tuple[str, str], dict[str, dict]] = {}
    for s in summaries:
        key = (s["dataset"], s.get("model", "?"))
        by_dm.setdefault(key, {})[s.get("kind", "?")] = s

    lines: list[str] = ["# Cross-method analysis", ""]
    if not summaries:
        lines.append("_No bnb_/gguf_ summaries found in results/._")
        report = "\n".join(lines)
        (results_dir / "analysis_report.md").write_text(report, encoding="utf-8")
        return report

    for (dataset, model), methods in sorted(by_dm.items()):
        lines += [f"## {dataset}  (model={model})", ""]

        # one table per method
        for kind, s in sorted(methods.items()):
            bits = sorted((int(b) for b in s["bits"]), reverse=True)
            lines.append(f"**{kind}** — bit-widths {bits}")
            header = "| metric | " + " | ".join(f"{b}-bit" for b in bits) + " |"
            lines.append(header)
            lines.append("|" + "---|" * (len(bits) + 1))
            for m in HEADLINE:
                row = " | ".join(_fmt(s["results"][str(b)].get(m)) for b in bits)
                lines.append(f"| {m} | {row} |")
            cliff = s.get("cliff_bits")
            sdi = s.get("selective_damage", {}).get("selective_damage_index")
            dec = s.get("decoupling", {}).get("decoupling_index")
            lines += [
                "",
                f"- over-confidence cliff bit-width: **{cliff}**",
                f"- selective-damage index (hi→lo): {_fmt(sdi)}  |  "
                f"decoupling index: {_fmt(dec)}",
                "",
            ]

        # overlay plots across methods for this dataset
        for metric, ylabel in (
            ("over_confident_wrong_rate", "over-confident-wrong rate"),
            ("accuracy_on_answered", "accuracy on answered"),
            ("abstention_f1", "abstention F1"),
            ("ece", "ECE"),
        ):
            series = {kind: _series(s, metric) for kind, s in methods.items()}
            out = overlay_vs_bits(
                series, fig_dir / f"cmp_{dataset}_{model}_{metric}.png",
                title=f"{model} on {dataset}: {ylabel} vs bit-width",
                ylabel=ylabel,
            )
            lines.append(f"![{metric}]({out.as_posix()})")
        lines.append("")

    report = "\n".join(lines)
    (results_dir / "analysis_report.md").write_text(report, encoding="utf-8")
    return report


def significance_report(results_dir: str | Path = "results", *, signal: str = "token_prob",
                        n_boot: int = 2000) -> str:
    """Paired-bootstrap significance of hi-bit -> lo-bit change per method/dataset."""
    from ..analysis.stats import benjamini_hochberg, paired_bootstrap_diff  # noqa: PLC0415
    from ..eval.pipeline import read_records  # noqa: PLC0415

    results_dir = Path(results_dir)
    lines = ["# Significance (paired bootstrap, hi-bit vs lo-bit)", ""]
    collected = []  # (label, metric, diff, p) across ALL tests, for FDR control
    for s in load_summaries(results_dir):
        bits = sorted((int(b) for b in s["bits"]), reverse=True)
        hi, lo = bits[0], bits[-1]
        try:
            hi_recs = read_records(s["results"][str(hi)]["path"])
            lo_recs = read_records(s["results"][str(lo)]["path"])
        except Exception as e:
            lines.append(f"- {s.get('kind')} {s['dataset']}: raw records missing ({e})")
            continue
        label = f"{s.get('kind')}·{s['dataset']}·{s.get('model')} {hi}→{lo}"
        lines.append(f"## {label}, n_boot={n_boot}")
        for metric in ("abstention_f1", "over_confident_wrong_rate", "accuracy_on_answered"):
            r = paired_bootstrap_diff(hi_recs, lo_recs, metric, signal=signal, n_boot=n_boot)
            p = r["p_two_sided"]
            star = "*" if (p == p and p < 0.05) else " "
            lines.append(
                f"  {star} {metric:30s} d(lo-hi)={r['diff_lo_minus_hi']:+.3f} "
                f"95%CI[{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}] p={p:.3f} (n={r['n']})"
            )
            collected.append((label, metric, r["diff_lo_minus_hi"], p))
        lines.append("")

    # --- Multiple-comparison correction across every test above (BH-FDR) ---
    pvals = [c[3] for c in collected]
    qvals, reject = benjamini_hochberg(pvals, alpha=0.05)
    n_tests = sum(1 for p in pvals if p == p)
    n_raw = sum(1 for p in pvals if p == p and p < 0.05)
    n_surv = int(reject.sum())
    lines += [
        "# Multiple-comparison correction (Benjamini-Hochberg FDR, alpha=0.05)",
        "",
        f"{n_tests} tests total; {n_raw} significant at raw p<.05; "
        f"**{n_surv} survive FDR**.",
        "",
        "| test | metric | d(lo-hi) | raw p | BH q | survives |",
        "|---|---|---|---|---|---|",
    ]
    rows = sorted(zip(collected, qvals, reject, strict=False), key=lambda t: (t[0][3]))
    for (label, metric, diff, p), q, rej in rows:
        if p != p:
            continue
        lines.append(f"| {label} | {metric} | {diff:+.3f} | {p:.3f} | "
                     f"{q:.3f} | {'YES' if rej else 'no'} |")
    report = "\n".join(lines)
    (results_dir / "significance_report.md").write_text(report, encoding="utf-8")
    return report


def controls_report(results_dir: str | Path = "results") -> str:
    """Confound controls: instruction-following (format), degeneration, length.

    Computed retroactively from each record's output_text, so it works on
    already-collected raw records. If format-compliance stays high and
    degeneration stays low while abstention F1 falls, the abstention drop is NOT
    explained by instruction-following / capability collapse.
    """
    from ..eval.pipeline import read_records  # noqa: PLC0415
    from ..eval.quality import is_degenerate, n_words  # noqa: PLC0415
    from ..eval.scoring import score_run  # noqa: PLC0415

    results_dir = Path(results_dir)
    lines = ["# Confound controls (instruction-following / degeneration / length)", ""]
    for s in load_summaries(results_dir):
        lines.append(f"## {s.get('kind')} · {s['dataset']} ({s.get('model')})")
        lines.append("| bit | abstention F1 | format-ok % | degenerate % | mean words |")
        lines.append("|---|---|---|---|---|")
        for b in sorted((int(x) for x in s["bits"]), reverse=True):
            path = s["results"][str(b)]["path"]
            try:
                recs = read_records(path)
                m = score_run(path)
            except Exception:
                continue
            if not recs:
                continue
            fmt = sum(1 for r in recs if r.get("verbalized_confidence") is not None) / len(recs)
            deg = sum(1 for r in recs if is_degenerate(r.get("output_text", ""))) / len(recs)
            wlen = sum(n_words(r.get("output_text", "")) for r in recs) / len(recs)
            lines.append(f"| {b} | {_fmt(m.get('abstention_f1'))} | {100 * fmt:.0f}% "
                         f"| {100 * deg:.0f}% | {wlen:.0f} |")
        lines.append("")
    report = "\n".join(lines)
    (results_dir / "controls_report.md").write_text(report, encoding="utf-8")
    return report


def signals_report(results_dir: str | Path = "results") -> str:
    """Compare token-prob vs verbalized confidence (AUROC/ECE) per bit (H2 signal choice)."""
    from ..eval.scoring import score_run  # noqa: PLC0415

    results_dir = Path(results_dir)
    lines = ["# Confidence-signal comparison (token_prob vs verbalized)", ""]
    for s in load_summaries(results_dir):
        lines.append(f"## {s.get('kind')} · {s['dataset']} ({s.get('model')})")
        lines.append("| bit | AUROC(tok) | AUROC(verb) | ECE(tok) | ECE(verb) |")
        lines.append("|---|---|---|---|---|")
        for b in sorted((int(x) for x in s["bits"]), reverse=True):
            path = s["results"][str(b)]["path"]
            try:
                tp = score_run(path, signal="token_prob")
                vb = score_run(path, signal="verbalized")
            except Exception:
                continue
            lines.append(
                f"| {b} | {_fmt(tp.get('auroc_conf_vs_correct'))} "
                f"| {_fmt(vb.get('auroc_conf_vs_correct'))} "
                f"| {_fmt(tp.get('ece'))} | {_fmt(vb.get('ece'))} |"
            )
        lines.append("")
    report = "\n".join(lines)
    (results_dir / "signals_report.md").write_text(report, encoding="utf-8")
    return report
