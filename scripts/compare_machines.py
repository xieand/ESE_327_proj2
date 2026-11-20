"""Overview figure comparing GPU speedups across machines.

usage: python scripts/compare_machines.py results/rtx3060ti results/jetson-nano-orin [--out results/overview.png]

For every classifier and table, plots the speedup CPU time / GPU time
(training + prediction per fold) measured on each machine. The decision tree
is the information-gain configuration, as in the per-machine speedup figure.
Exhaustive kNN on the three largest tables is compared on the holdout split
that both devices ran (see benchmark.py).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from benchmark import COLORS, MAIN, MAIN_LABEL, Table, load, total_ms


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("machines", nargs="+", help="result folders containing bench.csv")
    ap.add_argument("--out", default="results/overview.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = {Path(m).name: Table(load(Path(m) / "bench.csv")) for m in args.machines}
    first = next(iter(data.values()))
    datasets = first.datasets          # smallest at the bottom, largest at the top
    n_m = len(data)
    height = 0.8 / n_m
    shades = [1.0, 0.5, 0.3]

    fig, axes = plt.subplots(1, len(MAIN), figsize=(4.1 * len(MAIN), 0.42 * len(datasets) + 1.8), sharey=True)
    for ax, config in zip(axes, MAIN):
        for k, (machine, t) in enumerate(data.items()):
            ys, vals, holdout = [], [], []
            for i, d in enumerate(datasets):
                p = t.pair(d, config)
                if not p:
                    continue
                ys.append(i - (k - (n_m - 1) / 2) * height)
                vals.append(total_ms(p[0]) / total_ms(p[1]))
                holdout.append(p[0]["split"] == "holdout")
            bars = ax.barh(ys, vals, height=height, color=COLORS[config], alpha=shades[k % len(shades)],
                           label=machine)
            for b, v, h in zip(bars, vals, holdout):
                text = (f"{v:.0f}" if v >= 10 else f"{v:.1f}" if v >= 1 else f"{v:.2f}") + "×" + ("*" if h else "")
                ax.text(v * 1.08, b.get_y() + b.get_height() / 2, text, va="center", fontsize=7)
        ax.axvline(1.0, color="k", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlim(0.008, 250)
        ticks = [0.01, 0.1, 1, 10, 100]
        ax.set_xticks(ticks, [f"{t:g}×" for t in ticks])
        ax.minorticks_off()
        ax.set_title(MAIN_LABEL[config])
        ax.set_xlabel("CPU time / GPU time")
        ax.grid(True, axis="x", which="both", alpha=0.3)
        ax.legend(loc="lower right", fontsize=7)
    axes[0].set_yticks(range(len(datasets)), [f"{d}  ({first.records(d):,})" for d in datasets], fontsize=8)
    fig.suptitle("GPU speedup of each classifier on each machine: training + prediction per fold "
                 "(log scale, >1 means GPU faster; *: holdout split)")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
