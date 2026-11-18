"""Benchmark sweep: every classifier on every UCI table, CPU and GPU.

usage: python scripts/benchmark.py [--datasets mushroom adult ...] [--configs dtree_gain knn ...]
                                   [--devices cpu gpu] [--repeat 3] [--out results]

Classifier configurations
  dtree_gain, dtree_gain_ratio, dtree_gini        decision tree, unpruned
  dtree_*_pruned                                  ... with pessimistic pruning (--prune)
  nbayes                                          naive Bayes
  knn                                             k-nearest neighbors, k = 5
  logreg                                          multinomial logistic regression

Every run is a stratified 10-fold cross-validation with the same folds for
every classifier. The one exception is exhaustive k-nearest neighbors on the
three largest tables, which takes minutes per fold on the CPU: there the CPU
and the GPU are timed on one stratified holdout split, and the GPU
additionally runs the 10 folds, which is where the reported accuracy comes
from (the two builds give identical predictions).

Writes <out>/bench.csv, <out>/summary.md and the figures accuracy.png,
speedup.png, scaling.png, memory.png and tree_measures.png. Times are means
per fold (training on 90% of the table, predicting 10%); a run that takes
less than 20 s is repeated and the median is kept.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Smallest to largest.
DATASETS = ["votes", "tictactoe", "splice", "chess", "mushroom", "nursery", "letter", "krkopt", "adult",
            "connect4", "census", "covertype", "poker"]

# id -> (label, algorithm, options)
CONFIGS = {
    "dtree_gain": ("tree: gain", "dtree", ["--measure", "gain"]),
    "dtree_gain_pruned": ("tree: gain, pruned", "dtree", ["--measure", "gain", "--prune"]),
    "dtree_gain_ratio": ("tree: gain ratio", "dtree", ["--measure", "gain_ratio"]),
    "dtree_gain_ratio_pruned": ("tree: gain ratio, pruned", "dtree", ["--measure", "gain_ratio", "--prune"]),
    "dtree_gini": ("tree: Gini", "dtree", ["--measure", "gini"]),
    "dtree_gini_pruned": ("tree: Gini, pruned", "dtree", ["--measure", "gini", "--prune"]),
    "nbayes": ("naive Bayes", "nbayes", []),
    "knn": ("kNN (k=5)", "knn", ["--k", "5"]),
    "logreg": ("logistic regression", "logreg", []),
}
# The four algorithms as compared across devices (one tree configuration).
MAIN = ["dtree_gain", "nbayes", "knn", "logreg"]
MAIN_LABEL = {"dtree_gain": "decision tree (gain)", "nbayes": "naive Bayes", "knn": "kNN (k=5)",
              "logreg": "logistic regression"}
# Exhaustive kNN on the CPU: one holdout split instead of 10 folds.
KNN_HOLDOUT = {"census": "0.1", "covertype": "0.1", "poker": "0.05"}

FIELDS = ["dataset", "records", "attributes", "classes", "config", "algorithm", "device", "split", "folds",
          "accuracy", "accuracy_std", "macro_f1", "train_ms", "predict_ms", "read_ms", "peak_host_mb",
          "peak_device_mb", "tree_nodes", "tree_leaves", "tree_depth", "iterations", "converged", "options",
          "device_name"]
TEXT_FIELDS = {"dataset", "config", "algorithm", "device", "split", "options", "device_name"}


def run_once(binary: Path, table: Path, args: list[str], timeout: float) -> tuple[dict, float]:
    start = time.perf_counter()
    proc = subprocess.run([str(binary), str(table), "--json"] + args, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip())
    return json.loads(proc.stdout.strip().splitlines()[-1]), time.perf_counter() - start


def measure(binary: Path, table: Path, args: list[str], repeat: int, timeout: float) -> dict:
    first, seconds = run_once(binary, table, args, timeout)
    runs = [first]
    if seconds < 20:
        runs += [run_once(binary, table, args, timeout)[0] for _ in range(repeat - 1)]
    r = dict(first)
    for k in ("train_ms", "predict_ms", "read_ms"):
        r[k] = statistics.median(x[k] for x in runs)
    return r


def load(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in FIELDS:
            if k not in TEXT_FIELDS:
                r[k] = float(r[k]) if r[k] != "" else None
    return rows


class Table:
    """Lookup of the measured rows."""

    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.datasets = [d for d in DATASETS if any(r["dataset"] == d for r in rows)]
        self.datasets += sorted({r["dataset"] for r in rows} - set(DATASETS))
        self.configs = [c for c in CONFIGS if any(r["config"] == c for r in rows)]
        self.main = [c for c in MAIN if c in self.configs]

    def get(self, dataset: str, config: str, device: str, split: str = "kfold") -> dict | None:
        for r in self.rows:
            if (r["dataset"], r["config"], r["device"], r["split"]) == (dataset, config, device, split):
                return r
        return None

    def records(self, dataset: str) -> int:
        return int(next(r["records"] for r in self.rows if r["dataset"] == dataset))

    def accuracy(self, dataset: str, config: str) -> dict | None:
        """The 10-fold result; CPU and GPU agree, so either will do."""
        return self.get(dataset, config, "cpu") or self.get(dataset, config, "gpu")

    def pair(self, dataset: str, config: str) -> tuple[dict, dict] | None:
        """CPU and GPU rows measured on the same split."""
        for split in ("kfold", "holdout"):
            cpu, gpu = self.get(dataset, config, "cpu", split), self.get(dataset, config, "gpu", split)
            if cpu and gpu:
                return cpu, gpu
        return None


def total_ms(r: dict) -> float:
    return r["train_ms"] + r["predict_ms"]


def fmt_ms(v: float) -> str:
    return f"{v:,.0f}" if v >= 100 else f"{v:.1f}" if v >= 10 else f"{v:.2f}"


def fmt_x(v: float) -> str:
    return f"{v:.0f}x" if v >= 10 else f"{v:.1f}x" if v >= 1 else f"{v:.2f}x"


# -------------------------------------------------------------------- summary

def summary(t: Table, out: Path) -> None:
    lines: list[str] = []
    device = next((r["device_name"] for r in t.rows if r["device"] == "gpu"), "")
    lines += ["# Benchmark summary", ""]
    if device:
        lines += [f"GPU: {device}", ""]

    lines += ["## Accuracy (%), stratified 10-fold cross-validation", ""]
    lines.append("| dataset | records | " + " | ".join(CONFIGS[c][0] for c in t.configs) + " |")
    lines.append("|---|---|" + "---|" * len(t.configs))
    for d in t.datasets:
        cells = [t.accuracy(d, c) for c in t.configs]
        best = max((r["accuracy"] for r in cells if r), default=0)
        text = []
        for r in cells:
            if not r:
                text.append("-")
            else:
                s = f"{100 * r['accuracy']:.1f}"
                text.append(f"**{s}**" if r["accuracy"] >= best - 5e-4 else s)
        lines.append(f"| {d} | {t.records(d):,} | " + " | ".join(text) + " |")

    lines += ["", "## Time per fold (ms): training + prediction, CPU vs GPU", "",
              "Mean over the folds. *: one stratified holdout split instead of 10 folds.", ""]
    head = "| dataset |"
    for c in t.main:
        head += f" {MAIN_LABEL[c]} CPU | GPU | speedup |"
    lines += [head, "|---|" + "---|---|---|" * len(t.main)]
    for d in t.datasets:
        row = f"| {d} |"
        for c in t.main:
            p = t.pair(d, c)
            if not p:
                row += " - | - | - |"
                continue
            cpu, gpu = p
            star = "*" if cpu["split"] == "holdout" else ""
            row += f" {fmt_ms(total_ms(cpu))}{star} | {fmt_ms(total_ms(gpu))}{star} | {fmt_x(total_ms(cpu) / total_ms(gpu))} |"
        lines.append(row)

    lines += ["", "## Training and prediction time per fold (ms)", ""]
    lines.append("| dataset | classifier | split | CPU train | CPU predict | GPU train | GPU predict |")
    lines.append("|---|---|---|---|---|---|---|")
    for d in t.datasets:
        for c in t.configs:
            for split in ("kfold", "holdout"):
                cpu, gpu = t.get(d, c, "cpu", split), t.get(d, c, "gpu", split)
                if not cpu and not gpu:
                    continue
                cells = [fmt_ms(r[k]) if r else "-" for r in (cpu, gpu) for k in ("train_ms", "predict_ms")]
                lines.append(f"| {d} | {CONFIGS[c][0]} | {'10-fold' if split == 'kfold' else 'holdout'} | "
                             + " | ".join(cells) + " |")

    lines += ["", "## Peak memory (MB)", "",
              "Host: peak resident memory of the CPU program (includes the loaded table). "
              "Device: peak GPU memory allocated by the CUDA program.", ""]
    head = "| dataset |"
    for c in t.main:
        head += f" {MAIN_LABEL[c]} host | device |"
    lines += [head, "|---|" + "---|---|" * len(t.main)]
    for d in t.datasets:
        row = f"| {d} |"
        for c in t.main:
            cpu = t.get(d, c, "cpu") or t.get(d, c, "cpu", "holdout")
            gpu = t.get(d, c, "gpu") or t.get(d, c, "gpu", "holdout")
            row += f" {cpu['peak_host_mb']:.0f} |" if cpu else " - |"
            row += f" {gpu['peak_device_mb']:.0f} |" if gpu else " - |"
        lines.append(row)

    trees = [c for c in t.configs if c.startswith("dtree")]
    if trees:
        lines += ["", "## Decision trees: size and training time", "",
                  "Nodes and depth are means over the folds; time is CPU training per fold.", ""]
        lines.append("| dataset | measure | accuracy % | nodes | depth | pruned accuracy % | pruned nodes | CPU train ms | GPU train ms |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for d in t.datasets:
            for c in trees:
                if c.endswith("_pruned"):
                    continue
                r, p = t.accuracy(d, c), t.accuracy(d, c + "_pruned")
                if not r:
                    continue
                cpu, gpu = t.get(d, c, "cpu"), t.get(d, c, "gpu")
                lines.append(
                    f"| {d} | {CONFIGS[c][0].split(': ')[1]} | {100 * r['accuracy']:.1f} | {r['tree_nodes']:,.0f} | "
                    f"{r['tree_depth']:.0f} | " + (f"{100 * p['accuracy']:.1f} | {p['tree_nodes']:,.0f} | " if p else "- | - | ")
                    + (fmt_ms(cpu["train_ms"]) if cpu else "-") + " | " + (fmt_ms(gpu["train_ms"]) if gpu else "-") + " |")

    lr = [(d, t.accuracy(d, "logreg")) for d in t.datasets]
    if any(r for _, r in lr):
        lines += ["", "## Logistic regression: convergence", "",
                  "L-BFGS iterations (limit 100) and the share of folds that reached the gradient tolerance.", "",
                  "| dataset | iterations | converged folds |", "|---|---|---|"]
        for d, r in lr:
            if r:
                lines.append(f"| {d} | {r['iterations']:.0f} | {100 * r['converged']:.0f}% |")

    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out / 'summary.md'}")


# ---------------------------------------------------------------------- plots

COLORS = {"dtree_gain": "#2a7fb8", "nbayes": "#e08a1e", "knn": "#3a9e5c", "logreg": "#c0392b"}


def plot(t: Table, out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib / numpy not installed; skipping plots")
        return
    ds = t.datasets
    ds_labels = [f"{d}\n{t.records(d):,}" for d in ds]

    # ---- accuracy heat map
    acc = np.full((len(ds), len(t.configs)), np.nan)
    for i, d in enumerate(ds):
        for j, c in enumerate(t.configs):
            r = t.accuracy(d, c)
            if r:
                acc[i, j] = 100 * r["accuracy"]
    fig, ax = plt.subplots(figsize=(1.25 * len(t.configs) + 2.2, 0.48 * len(ds) + 1.8))
    # Color by the distance to the best classifier of the row, so that easy
    # and hard tables are both readable.
    gap = acc - np.nanmax(acc, axis=1, keepdims=True)
    im = ax.imshow(gap, cmap="RdYlGn", vmin=-25, vmax=0, aspect="auto")
    for i in range(len(ds)):
        for j in range(len(t.configs)):
            if not np.isnan(acc[i, j]):
                best = gap[i, j] > -0.05
                ax.text(j, i, f"{acc[i, j]:.1f}", ha="center", va="center", fontsize=9,
                        fontweight="bold" if best else "normal")
    ax.set_xticks(range(len(t.configs)), [CONFIGS[c][0].replace(": ", "\n").replace(", ", "\n") for c in t.configs],
                  fontsize=8)
    ax.set_yticks(range(len(ds)), [f"{d}  ({t.records(d):,})" for d in ds], fontsize=9)
    ax.xaxis.tick_top()
    fig.colorbar(im, ax=ax, label="points below the best classifier on the table", shrink=0.8)
    ax.set_title("Accuracy (%), stratified 10-fold cross-validation", pad=42)
    fig.tight_layout()
    fig.savefig(out / "accuracy.png", dpi=140)
    plt.close(fig)

    # ---- GPU speedup
    if any(t.pair(d, c) for d in ds for c in t.main):
        fig, ax = plt.subplots(figsize=(13, 4.6))
        width = 0.8 / len(t.main)
        for j, c in enumerate(t.main):
            xs, ys = [], []
            for i, d in enumerate(ds):
                p = t.pair(d, c)
                if p:
                    xs.append(i + (j - (len(t.main) - 1) / 2) * width)
                    ys.append(total_ms(p[0]) / total_ms(p[1]))
            ax.bar(xs, ys, width * 0.92, label=MAIN_LABEL[c], color=COLORS[c])
            for x, y in zip(xs, ys):
                ax.text(x, y * 1.06, f"{y:.0f}" if y >= 10 else f"{y:.1f}" if y >= 1 else f"{y:.2f}", ha="center", va="bottom", fontsize=6.5)
        ax.axhline(1.0, color="black", lw=0.9)
        ax.set_yscale("log")
        ax.set_xticks(range(len(ds)), ds_labels, fontsize=8)
        ax.set_ylabel("CPU time / GPU time")
        ax.set_title("GPU speedup, training + prediction per fold (tables ordered by size; above 1 the GPU is faster)")
        ax.grid(True, axis="y", which="both", alpha=0.3)
        ax.legend(fontsize=8, ncol=4, loc="upper left")
        fig.tight_layout()
        fig.savefig(out / "speedup.png", dpi=140)
        plt.close(fig)

        # ---- time against table size
        fig, axes = plt.subplots(1, len(t.main), figsize=(4.4 * len(t.main), 4.0), squeeze=False)
        for ax, c in zip(axes[0], t.main):
            pts = [(t.records(d), p) for d in ds if (p := t.pair(d, c))]
            pts.sort(key=lambda x: x[0])
            ax.plot([n for n, _ in pts], [total_ms(p[0]) for _, p in pts], "o-", color="#555555", label="CPU")
            ax.plot([n for n, _ in pts], [total_ms(p[1]) for _, p in pts], "s-", color=COLORS[c], label="GPU")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(MAIN_LABEL[c])
            ax.set_xlabel("records in the table")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=8)
        axes[0][0].set_ylabel("training + prediction per fold (ms)")
        fig.tight_layout()
        fig.savefig(out / "scaling.png", dpi=140)
        plt.close(fig)

    # ---- memory
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    width = 0.8 / max(1, len(t.main))
    for ax, device, key, title in ((axes[0], "cpu", "peak_host_mb", "CPU programs: peak host memory"),
                                   (axes[1], "gpu", "peak_device_mb", "GPU programs: peak device memory")):
        for j, c in enumerate(t.main):
            xs, ys = [], []
            for i, d in enumerate(ds):
                r = t.get(d, c, device) or t.get(d, c, device, "holdout")
                if r:
                    xs.append(i + (j - (len(t.main) - 1) / 2) * width)
                    ys.append(max(r[key], 0.01))
            ax.bar(xs, ys, width * 0.92, label=MAIN_LABEL[c], color=COLORS[c])
        ax.set_yscale("log")
        ax.set_xticks(range(len(ds)), ds, fontsize=8, rotation=35, ha="right")
        ax.set_ylabel("MB")
        ax.set_title(title)
        ax.grid(True, axis="y", which="both", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "memory.png", dpi=140)
    plt.close(fig)

    # ---- the three attribute selection measures
    measures = [("dtree_gain", "gain", "#2a7fb8"), ("dtree_gain_ratio", "gain ratio", "#8e5bb5"),
                ("dtree_gini", "Gini", "#d4832f")]
    measures = [m for m in measures if m[0] in t.configs]
    if measures:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
        width = 0.8 / len(measures)
        for j, (c, label, color) in enumerate(measures):
            off = (j - (len(measures) - 1) / 2) * width
            for i, d in enumerate(ds):
                r, p = t.accuracy(d, c), t.accuracy(d, c + "_pruned")
                cpu = t.get(d, c, "cpu")
                if not r:
                    continue
                first = i == 0
                axes[0].bar(i + off, 100 * r["accuracy"], width * 0.92, color=color, label=label if first else None)
                if p:
                    axes[0].plot([i + off], [100 * p["accuracy"]], "k_", ms=9, mew=1.8,
                                 label="pruned" if first and j == 0 else None)
                axes[1].bar(i + off, r["tree_nodes"], width * 0.92, color=color, label=label if first else None)
                if p:
                    axes[1].plot([i + off], [p["tree_nodes"]], "k_", ms=9, mew=1.8,
                                 label="pruned" if first and j == 0 else None)
                if cpu:
                    axes[2].bar(i + off, cpu["train_ms"], width * 0.92, color=color, label=label if first else None)
        axes[0].set_ylim(40, 101)
        axes[0].set_title("accuracy (%), 10-fold")
        axes[1].set_yscale("log")
        axes[1].set_title("tree nodes")
        axes[2].set_yscale("log")
        axes[2].set_title("CPU training time per fold (ms)")
        for ax in axes:
            ax.set_xticks(range(len(ds)), ds, fontsize=8, rotation=35, ha="right")
            ax.grid(True, axis="y", which="both", alpha=0.3)
            if ax is axes[0]:
                ax.legend(fontsize=8, ncol=4, loc="lower center", framealpha=0.95)
            else:
                ax.legend(fontsize=8, loc="upper left")
        fig.suptitle("Decision tree: the three attribute selection measures (bars: unpruned; marks: pruned)")
        fig.tight_layout()
        fig.savefig(out / "tree_measures.png", dpi=140)
        plt.close(fig)
    print(f"wrote figures to {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS), metavar="CONFIG")
    ap.add_argument("--devices", nargs="+", default=["cpu", "gpu"], choices=["cpu", "gpu", "gpu_emu"])
    ap.add_argument("--bin-dir", default=str(ROOT / "bin"))
    ap.add_argument("--repeat", type=int, default=3, help="runs per point when a run is short; the median time is kept")
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--plot-only", action="store_true", help="redraw figures and summary from <out>/bench.csv")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        t = Table(load(out / "bench.csv"))
        summary(t, out)
        plot(t, out)
        return

    exe = ".exe" if os.name == "nt" else ""
    rows: list[dict] = []
    for ds in args.datasets:
        table = ROOT / "data" / "uci" / f"{ds}.csv"
        if not table.exists():
            print(f"{table} missing; run scripts/fetch_uci.py {ds}")
            continue
        for config in args.configs:
            label, algorithm, options = CONFIGS[config]
            for device in args.devices:
                binary = Path(args.bin_dir) / f"{algorithm}_{device}{exe}"
                if not binary.exists():
                    print(f"{binary} not built; skipping")
                    continue
                # kNN on the largest tables: holdout on every device, and the
                # 10 folds only where they take seconds rather than hours.
                splits = [["--folds", "10"]]
                if config == "knn" and ds in KNN_HOLDOUT:
                    splits = [["--holdout", KNN_HOLDOUT[ds]]] + (splits if device == "gpu" else [])
                for split in splits:
                    try:
                        r = measure(binary, table, options + split, args.repeat, args.timeout)
                    except (subprocess.TimeoutExpired, RuntimeError) as e:
                        print(f"{ds} {config} {device}: {type(e).__name__} {e}")
                        continue
                    r.update(config=config, device=device)
                    rows.append({k: r.get(k, "") for k in FIELDS})
                    print(f"{ds:10s} {config:24s} {device:4s} {r['split']:8s} acc {r['accuracy']:.4f}  "
                          f"train {r['train_ms']:10.2f} ms  predict {r['predict_ms']:10.2f} ms  "
                          f"host {r['peak_host_mb']:.0f} MB  device {r['peak_device_mb']:.0f} MB", flush=True)
                    with open(out / "bench.csv", "w", newline="", encoding="utf-8") as f:
                        w = csv.DictWriter(f, fieldnames=FIELDS)
                        w.writeheader()
                        w.writerows(rows)
    print(f"wrote {out / 'bench.csv'}")
    if rows:
        t = Table(load(out / "bench.csv"))
        summary(t, out)
        plot(t, out)


if __name__ == "__main__":
    main()
