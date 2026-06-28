"""Reliability diagrams + degradation curves (workflow §3 Stage 4).

matplotlib is imported lazily with the non-interactive Agg backend so importing
the package never requires a display.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _plt():
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    return plt


def degradation_curve(
    bits: list[float],
    series: dict[str, list[float]],
    out_path: str | Path,
    *,
    title: str = "Degradation vs bit-width",
    xlabel: str = "bit-width (lower precision ->)",
) -> Path:
    """Plot one or more metrics against bit-width on a shared axis.

    ``series`` maps metric label -> values aligned with ``bits``. Bits are shown
    high-precision (left) to low-precision (right) to read as a degradation curve.
    """
    plt = _plt()
    order = sorted(range(len(bits)), key=lambda i: bits[i], reverse=True)
    x = [bits[i] for i in order]
    xpos = list(range(len(x)))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, vals in series.items():
        y = [vals[i] for i in order]
        ax.plot(xpos, y, marker="o", label=label)
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"{int(b) if float(b).is_integer() else b}-bit" for b in x])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("metric value")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def overlay_vs_bits(
    series_by_label: dict[str, dict[int, float]],
    out_path: str | Path,
    *,
    title: str,
    ylabel: str,
) -> Path:
    """Overlay one metric vs bit-width for multiple methods on a shared axis.

    ``series_by_label`` maps method label -> {bits: value}. Different methods may
    cover different bit sets; the x axis is the union, ordered high->low precision.
    """
    plt = _plt()
    all_bits = sorted({b for s in series_by_label.values() for b in s}, reverse=True)
    xpos = {b: i for i, b in enumerate(all_bits)}
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for label, s in series_by_label.items():
        bits = sorted(s.keys(), reverse=True)
        ax.plot([xpos[b] for b in bits], [s[b] for b in bits], marker="o", label=label)
    ax.set_xticks(list(xpos.values()))
    ax.set_xticklabels([f"{b}-bit" for b in all_bits])
    ax.set_xlabel("bit-width (lower precision ->)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def reliability_diagram(curve: dict[str, Any], out_path: str | Path) -> Path:
    """Plot a reliability diagram from metrics.reliability_curve output."""
    plt = _plt()
    conf = curve["bin_confidence"]
    acc = curve["bin_accuracy"]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    ax.plot(conf, acc, marker="o", label="model")
    ax.set_xlabel("mean confidence")
    ax.set_ylabel("accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Reliability diagram")
    ax.legend()
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
