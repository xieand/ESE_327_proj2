"""Checks the shared C++ infrastructure (table loader, stratified splits,
metrics, evaluation loop) against pandas / scikit-learn.

For every table in data/examples and data/uci, bin/dataset_info is run and
compared with an independent reading of the same file:

  loader   record count, class counts, per-attribute value counts (categorical)
           or sum / min / max (numeric), missing-value counts
  k-fold   every record in exactly one fold, fold sizes within 1 of each other,
           every class spread over the folds within 1 of evenly, same seed ->
           same folds, another seed -> different folds
  holdout  the test set holds round(fraction * class size) records of each class
  metrics  majority-baseline accuracy per fold and pooled, and pooled macro-F1
           (scikit-learn f1_score), recomputed from the dumped fold ids

usage: python scripts/check_data.py [--bin-dir bin] [names...]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
FOLDS = 10
HOLDOUT = 0.3


class Results:
    def __init__(self) -> None:
        self.passed = 0
        self.failed: list[str] = []

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        if ok:
            self.passed += 1
        else:
            self.failed.append(name)
            print(f"  FAIL {name} {detail}")


def run(binary: Path, table: Path, *extra: str) -> dict:
    proc = subprocess.run([str(binary), str(table), "--json", *extra], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{binary.name} {table.name} {' '.join(extra)} failed:\n{proc.stderr}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def read_folds(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=np.int64, ndmin=1)


def majority_predictions(y: np.ndarray, fold: np.ndarray, n_folds: int, order: list[str]) -> np.ndarray:
    """Per test fold, the most frequent training class (ties: first in `order`, the C++ class-id order)."""
    pred = np.empty(len(y), dtype=object)
    for f in range(n_folds):
        train = y[fold != f]
        counts = [(train == c).sum() for c in order]
        pred[fold == f] = order[int(np.argmax(counts))]
    return pred


def check_table(res: Results, binary: Path, table: Path, tmp: Path) -> None:
    name = table.stem
    df = pd.read_csv(table, dtype=str, keep_default_na=False)
    types = {c.rsplit(":", 1)[0]: c.rsplit(":", 1)[1] for c in df.columns}
    df.columns = list(types)
    class_col = next(c for c, t in types.items() if t == "class")
    y = df[class_col].to_numpy()
    n = len(df)
    n_folds = min(FOLDS, n)

    dump = tmp / "folds.txt"
    info = run(binary, table, "--folds", str(n_folds), "--dump-folds", str(dump))

    # ---- loader
    res.check(f"{name}: records", info["records"] == n, f"{info['records']} != {n}")
    res.check(f"{name}: class counts", info["classes"] == df[class_col].value_counts().to_dict())
    res.check(f"{name}: class order", list(info["classes"]) == list(pd.unique(y)))
    attrs = [c for c in types if c != class_col]
    res.check(f"{name}: attribute names", [a["name"] for a in info["attributes"]] == attrs)
    for a in info["attributes"]:
        col = df[a["name"]]
        missing = int((col == "?").sum())
        ok = a["type"] == types[a["name"]] and a["missing"] == missing
        if a["type"] == "cat":
            ok = ok and a["values"] == col.value_counts().to_dict() and list(a["values"]) == list(pd.unique(col))
        else:
            v = pd.to_numeric(col[col != "?"]).to_numpy(dtype=np.float64)
            ok = ok and math.isclose(a["sum"], v.sum(), rel_tol=1e-9, abs_tol=1e-9) \
                and a["min"] == v.min() and a["max"] == v.max()
        res.check(f"{name}: attribute {a['name']}", bool(ok))

    # ---- stratified k-fold
    fold = read_folds(dump)
    res.check(f"{name}: fold ids", len(fold) == n and fold.min() >= 0 and fold.max() < n_folds)
    sizes = np.bincount(fold, minlength=n_folds)
    res.check(f"{name}: fold sizes balanced", sizes.max() - sizes.min() <= 1, str(sizes))
    res.check(f"{name}: fold sizes reported",
              [f["test"] for f in info["folds"]] == sizes.tolist()
              and all(f["train"] + f["test"] == n for f in info["folds"]))
    stratified = True
    for c in pd.unique(y):
        per_fold = np.bincount(fold[y == c], minlength=n_folds)
        stratified &= per_fold.max() - per_fold.min() <= 1
    res.check(f"{name}: folds stratified", bool(stratified))

    run(binary, table, "--folds", str(n_folds), "--dump-folds", str(tmp / "again.txt"))
    res.check(f"{name}: folds reproducible", np.array_equal(fold, read_folds(tmp / "again.txt")))
    if n >= 100:
        run(binary, table, "--folds", str(n_folds), "--seed", "1", "--dump-folds", str(tmp / "other.txt"))
        res.check(f"{name}: folds depend on seed", not np.array_equal(fold, read_folds(tmp / "other.txt")))

    # ---- metrics through the evaluation loop (majority baseline)
    order = list(info["classes"])
    pred = majority_predictions(y, fold, n_folds, order)
    per_fold = [float((pred[fold == f] == y[fold == f]).mean()) for f in range(n_folds)]
    res.check(f"{name}: per-fold accuracy",
              all(abs(a - b["majority_accuracy"]) < 1e-6 for a, b in zip(per_fold, info["folds"])))
    res.check(f"{name}: pooled accuracy", abs(float((pred == y).mean()) - info["majority_accuracy"]) < 1e-6)
    f1 = f1_score(y, pred, average="macro", zero_division=0)
    res.check(f"{name}: macro-F1", abs(f1 - info["majority_macro_f1"]) < 1e-6, f"{f1} != {info['majority_macro_f1']}")

    # ---- stratified holdout
    if n >= 10:
        hold = run(binary, table, "--holdout", str(HOLDOUT), "--dump-folds", str(dump))
        is_test = read_folds(dump)
        ok = len(is_test) == n and set(np.unique(is_test)) <= {0, 1}
        for c in pd.unique(y):
            ok = ok and int(is_test[y == c].sum()) == math.floor(HOLDOUT * int((y == c).sum()) + 0.5)
        res.check(f"{name}: holdout stratified", bool(ok))
        res.check(f"{name}: holdout sizes",
                  hold["folds"][0]["test"] == int(is_test.sum()) and hold["folds"][0]["train"] == n - int(is_test.sum()))

    print(f"  {name:22s} {n:>9,} records  read {info['read_ms']:8.1f} ms  peak {info['peak_host_mb']:7.1f} MB")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin-dir", default=str(ROOT / "bin"))
    ap.add_argument("names", nargs="*", help="table names (default: every table found)")
    args = ap.parse_args()

    binary = Path(args.bin_dir) / ("dataset_info" + (".exe" if os.name == "nt" else ""))
    if not binary.exists():
        print(f"{binary} not built; run make (mingw32-make on Windows) first")
        return 2
    tables = sorted((ROOT / "data" / "examples").glob("*.csv")) + sorted((ROOT / "data" / "uci").glob("*.csv"))
    if args.names:
        tables = [t for t in tables if t.stem in args.names]
    if not (ROOT / "data" / "uci").exists():
        print("data/uci missing; run scripts/fetch_uci.py for the benchmark tables")

    res = Results()
    tmp = Path(tempfile.mkdtemp(prefix="ml_check_"))
    for t in tables:
        check_table(res, binary, t, tmp)

    # A malformed table must be rejected, not silently loaded.
    bad = tmp / "bad.csv"
    for label, text in [("short row", "a:cat,b:num,c:class\nx,1,p\ny,2\n"),
                        ("non-numeric value", "a:cat,b:num,c:class\nx,1,p\ny,two,q\n"),
                        ("no class column", "a:cat,b:num\nx,1\n"),
                        ("unknown type", "a:cat,b:int,c:class\nx,1,p\n")]:
        bad.write_text(text, encoding="utf-8")
        proc = subprocess.run([str(binary), str(bad)], capture_output=True, text=True)
        res.check(f"rejects {label}", proc.returncode == 1 and "error" in proc.stderr, proc.stderr.strip())

    print(f"\n{res.passed} passed, {len(res.failed)} failed")
    for f in res.failed:
        print("  failed:", f)
    return 1 if res.failed else 0


if __name__ == "__main__":
    sys.exit(main())
