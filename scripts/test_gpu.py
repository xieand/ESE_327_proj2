"""CUDA builds against the CPU builds.

test_dtree.py and test_baselines.py already check every build they find
(cpu, gpu, gpu_emu) against the independent references. This suite adds the
direct comparison, on more and larger tables and with the GPU-only options:

  decision tree        same tree (printed text), same root scores and same
                       cross-validation predictions, also when a level is
                       split into many passes (--hist-mb)
  naive Bayes          same predictions on categorical tables; on numeric
                       attributes the statistics differ in the last bits
                       (summation order), so near-identical predictions
  k-nearest neighbors  same predictions
  logistic regression  same objective value at the optimum and near-identical
                       predictions (summation order; device exp / log)

The emulation build (gpu_emu) runs the kernels serially, so it only gets the
smaller tables.

usage: python scripts/test_gpu.py [--bin-dir bin] [--quick]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_baselines import random_tables  # noqa: E402
from test_dtree import Results  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
UCI = ROOT / "data" / "uci"
EXAMPLES = ROOT / "data" / "examples"
MEASURES = ["gain", "gain_ratio", "gini"]


def binaries(bin_dir: Path, algorithm: str) -> tuple[Path | None, dict[str, Path]]:
    exe = ".exe" if os.name == "nt" else ""
    cpu = bin_dir / f"{algorithm}_cpu{exe}"
    gpus = {v: bin_dir / f"{algorithm}_{v}{exe}" for v in ("gpu_emu", "gpu")}
    return (cpu if cpu.exists() else None), {v: p for v, p in gpus.items() if p.exists()}


def run(binary: Path, table: Path, args: list[str]) -> str:
    cmd = [str(binary), str(table)] + args
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed ({proc.returncode}):\n{proc.stderr}")
    return proc.stdout


def evaluate(binary: Path, table: Path, args: list[str], tmp: Path) -> tuple[dict, list[str]]:
    """Run report (JSON) and the predicted label of every record."""
    pred = tmp / f"pred_{binary.stem}.txt"
    out = run(binary, table, args + ["--json", "--predictions", str(pred)])
    return json.loads(out.strip().splitlines()[-1]), pred.read_text().split("\n")[:-1]


def agreement(a: list[str], b: list[str]) -> float:
    tested = [(x, y) for x, y in zip(a, b) if x != "-" or y != "-"]
    return sum(x == y for x, y in tested) / max(1, len(tested))


def uci(names: list[str]) -> list[tuple[str, Path]]:
    return [(n, UCI / f"{n}.csv") for n in names if (UCI / f"{n}.csv").exists()]


# -------------------------------------------------------------- decision tree

def test_dtree(res: Results, cpu: Path, gpus: dict[str, Path], tmp: Path, quick: bool) -> None:
    print("\n[decision tree] same tree, scores and predictions as dtree_cpu")
    small = [(p.stem, p) for p in sorted(EXAMPLES.glob("*.csv"))]
    small += random_tables(tmp, 6 if quick else 18, seed=41)
    small += uci(["votes", "tictactoe", "mushroom", "chess", "splice", "nursery"])
    medium = uci(["letter", "krkopt", "adult"])
    large = uci(["connect4", "census", "covertype", "poker"])
    option_sets = [[], ["--prune"], ["--max-bins", "12", "--max-depth", "7", "--min-split", "10"]]

    for variant, gpu in gpus.items():
        # Whole-table trees and root scores, plus 3-fold predictions.
        tables = small + ([] if quick else medium)
        for name, table in tables:
            bad = []
            runs = 0
            for measure in MEASURES:
                for opts in option_sets:
                    for budget in ([], ["--hist-mb", "0.002"]):   # few nodes per pass
                        base = ["--measure", measure] + opts
                        want = run(cpu, table, base + ["--print-tree", "--scores", "--folds", "0", "--quiet"])
                        got = run(gpu, table, base + budget + ["--print-tree", "--scores", "--folds", "0", "--quiet"])
                        runs += 1
                        if got != want:
                            bad.append(f"tree {measure} {' '.join(opts + budget)}")
                    _, p_cpu = evaluate(cpu, table, base + ["--folds", "3"], tmp)
                    _, p_gpu = evaluate(gpu, table, base + ["--folds", "3"], tmp)
                    runs += 1
                    if p_cpu != p_gpu:
                        bad.append(f"predictions {measure} {' '.join(opts)}")
            res.check(f"{variant}: {name}", not bad, "; ".join(bad[:4]) if bad else f"{runs} runs identical")

        # Large tables: one holdout split, tree statistics and predictions.
        if variant != "gpu" or quick:
            continue
        for name, table in large:
            bad = []
            for measure in MEASURES:
                for opts in ([], ["--prune"]):
                    base = ["--measure", measure, "--holdout", "0.2"] + opts
                    r_cpu, p_cpu = evaluate(cpu, table, base, tmp)
                    r_gpu, p_gpu = evaluate(gpu, table, base + (["--hist-mb", "8"] if measure == "gini" else []), tmp)
                    same_stats = all(r_cpu[k] == r_gpu[k] for k in ("tree_nodes", "tree_leaves", "tree_depth"))
                    if p_cpu != p_gpu or not same_stats:
                        bad.append(f"{measure} {' '.join(opts)}")
            res.check(f"{variant}: {name} (holdout)", not bad, "; ".join(bad) if bad else "6 runs identical")


# ---------------------------------------------------------------- naive Bayes

def has_numeric(table: Path) -> bool:
    with open(table, encoding="utf-8") as f:
        return ":num" in f.readline()


def test_nbayes(res: Results, cpu: Path, gpus: dict[str, Path], tmp: Path, quick: bool) -> None:
    print("\n[naive Bayes] predictions vs nbayes_cpu")
    tables = [(p.stem, p) for p in sorted(EXAMPLES.glob("*.csv"))]
    tables += random_tables(tmp, 6 if quick else 18, seed=43)
    tables += uci(["votes", "tictactoe", "mushroom", "chess", "splice", "nursery", "letter", "krkopt", "adult"])
    if not quick:
        tables += uci(["connect4", "census", "covertype", "poker"])
    query = "youth,medium,yes,fair"
    for variant, gpu in gpus.items():
        example = EXAMPLES / "buys_computer.csv"
        res.check(f"{variant}: Example 8.4 query",
                  run(gpu, example, ["--query", query, "--folds", "0", "--quiet"]) ==
                  run(cpu, example, ["--query", query, "--folds", "0", "--quiet"]))
        for name, table in tables:
            for opts in ([], ["--alpha", "0"]):
                _, p_cpu = evaluate(cpu, table, ["--folds", "5"] + opts, tmp)
                _, p_gpu = evaluate(gpu, table, ["--folds", "5"] + opts, tmp)
                agree = agreement(p_cpu, p_gpu)
                label = f"{variant}: {name} {' '.join(opts)}".rstrip()
                if has_numeric(table):
                    res.check(label, agree >= 0.9995, f"agreement {agree:.6f}")
                else:
                    res.check(label, p_cpu == p_gpu, "identical" if p_cpu == p_gpu else f"agreement {agree:.6f}")


# -------------------------------------------------------- k-nearest neighbors

def test_knn(res: Results, cpu: Path, gpus: dict[str, Path], tmp: Path, quick: bool) -> None:
    print("\n[k-nearest neighbors] same predictions as knn_cpu")
    small = [(p.stem, p, ["--folds", "5"]) for p in sorted(EXAMPLES.glob("*.csv"))]
    small += [(n, p, ["--folds", "5"]) for n, p in random_tables(tmp, 6 if quick else 18, seed=47)]
    small += [(n, p, ["--folds", "5"]) for n, p in uci(["votes", "tictactoe", "mushroom", "chess", "splice"])]
    small += [(n, p, ["--holdout", "0.1"]) for n, p in uci(["nursery", "letter", "krkopt"])]
    large = [(n, p, ["--folds", "5"]) for n, p in uci(["nursery", "letter", "krkopt", "adult"])]
    large += [(n, p, ["--holdout", "0.05"]) for n, p in uci(["connect4"])]
    large += [(n, p, ["--holdout", "0.005"]) for n, p in uci(["census", "covertype", "poker"])]
    for variant, gpu in gpus.items():
        for name, table, split in small + (large if variant == "gpu" and not quick else []):
            bad = []
            for k in (1, 5, 10):
                _, p_cpu = evaluate(cpu, table, split + ["--k", str(k)], tmp)
                _, p_gpu = evaluate(gpu, table, split + ["--k", str(k)], tmp)
                if p_cpu != p_gpu:
                    bad.append(f"k={k}: agreement {agreement(p_cpu, p_gpu):.6f}")
            res.check(f"{variant}: {name} {' '.join(split)}", not bad, "; ".join(bad) if bad else "k = 1, 5, 10 identical")


# -------------------------------------------------------- logistic regression

def test_logreg(res: Results, cpu: Path, gpus: dict[str, Path], tmp: Path, quick: bool) -> None:
    print("\n[logistic regression] objective and predictions vs logreg_cpu")
    small = [(p.stem, p) for p in sorted(EXAMPLES.glob("*.csv"))]
    small += random_tables(tmp, 6 if quick else 12, seed=53, n_choices=[60, 200, 600])
    small += uci(["votes", "tictactoe", "mushroom", "chess"])
    medium = uci(["splice", "nursery", "letter", "adult"])
    large = uci(["connect4", "census", "covertype", "poker"])
    tight = ["--tol", "1e-7", "--max-iter", "3000", "--folds", "0"]
    for variant, gpu in gpus.items():
        # At the optimum both must have found the same minimum.
        for name, table in small:
            r_cpu, p_cpu = evaluate(cpu, table, tight, tmp)
            r_gpu, p_gpu = evaluate(gpu, table, tight, tmp)
            diff = abs(r_cpu["objective"] - r_gpu["objective"])
            agree = agreement(p_cpu, p_gpu)
            res.check(f"{variant}: {name} (optimum)", diff <= 1e-9 * max(1.0, abs(r_cpu["objective"])) and agree >= 0.995,
                      f"objective {r_gpu['objective']:.9f}, difference {diff:.1e}, agreement {agree:.4f}")
        # Default settings (100 iterations): the two runs take the same path
        # up to rounding, which the optimizer can amplify on the tables that
        # have not converged.
        tables = small + medium + (large if variant == "gpu" and not quick else [])
        for name, table in tables:
            split = ["--holdout", "0.2"] if (name, table) in large else ["--folds", "3"]
            r_cpu, p_cpu = evaluate(cpu, table, split, tmp)
            r_gpu, p_gpu = evaluate(gpu, table, split, tmp)
            diff = abs(r_cpu["objective"] - r_gpu["objective"])
            agree = agreement(p_cpu, p_gpu)
            res.check(f"{variant}: {name} {' '.join(split)}",
                      diff <= 1e-4 * max(1.0, abs(r_cpu["objective"])) and agree >= 0.99,
                      f"objective difference {diff:.1e}, agreement {agree:.4f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin-dir", default=str(ROOT / "bin"))
    ap.add_argument("--quick", action="store_true", help="fewer random tables, skip the larger UCI runs")
    ap.add_argument("--only", choices=["dtree", "nbayes", "knn", "logreg"], action="append",
                    help="test only this algorithm (repeatable)")
    args = ap.parse_args()

    res = Results()
    tmp = Path(tempfile.mkdtemp(prefix="gpu_tests_"))
    if not UCI.exists():
        print("data/uci missing: only the examples and random tables are tested (run scripts/fetch_uci.py)")
    found = False
    for algorithm, test in (("dtree", test_dtree), ("nbayes", test_nbayes), ("knn", test_knn), ("logreg", test_logreg)):
        if args.only and algorithm not in args.only:
            continue
        cpu, gpus = binaries(Path(args.bin_dir), algorithm)
        if cpu is None or not gpus:
            print(f"\n{algorithm}: needs {algorithm}_cpu and {algorithm}_gpu or {algorithm}_gpu_emu in {args.bin_dir}; skipped")
            continue
        found = True
        test(res, cpu, gpus, tmp, args.quick)
    if not found:
        print("\nno CUDA or emulation builds found (make emu / make gpu)")
        return 0

    print(f"\n{res.passed} passed, {len(res.failed)} failed")
    for f in res.failed:
        print("  failed:", f)
    return 1 if res.failed else 0


if __name__ == "__main__":
    sys.exit(main())
