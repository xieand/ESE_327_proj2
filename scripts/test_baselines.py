"""Correctness tests for naive Bayes, k-nearest neighbors and logistic regression.

  naive Bayes
    1. Han & Kamber, Example 8.4: P(X|C)P(C) for X = (youth, medium, yes, fair).
    2. Cross-validation predictions against scikit-learn (CategoricalNB for the
       categorical attributes + GaussianNB for the numeric ones, combined into
       one posterior). Every prediction must agree unless scikit-learn's two
       best classes are tied.
  k-nearest neighbors
    3. Cross-validation predictions identical to an exhaustive NumPy search
       that follows the same distance definition and voting rule.
    4. scikit-learn KNeighborsClassifier on real-valued data (no distance ties).
  logistic regression
    5. The trained weights reach the same optimum as scikit-learn's
       LogisticRegression: same objective value and same class probabilities.
    6. Cross-validation predictions with default settings agree with scikit-learn.

usage: python scripts/test_baselines.py [--bin-dir bin] [--quick]
Every binary found among {nbayes,knn,logreg}_{cpu,gpu,gpu_emu} is tested.
"""
from __future__ import annotations

import argparse
import os
import random
import re
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from test_dtree import Results, random_table, write_table

ROOT = Path(__file__).resolve().parent.parent
UCI = ROOT / "data" / "uci"
EXAMPLES = ROOT / "data" / "examples"


def find_binaries(bin_dir: Path, algorithm: str) -> dict[str, Path]:
    exe = ".exe" if os.name == "nt" else ""
    names = [f"{algorithm}_{v}" for v in ("cpu", "gpu", "gpu_emu")]
    return {n: bin_dir / f"{n}{exe}" for n in names if (bin_dir / f"{n}{exe}").exists()}


def run(binary: Path, table: Path, args: list[str]) -> str:
    cmd = [str(binary), str(table)] + args
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed ({proc.returncode}):\n{proc.stderr}")
    return proc.stdout


def cpp_cv(binary: Path, table: Path, opts: list[str], split: list[str], tmp: Path):
    """Fold id and predicted class code of every record ('-' = not tested -> -1)."""
    pred, fold = tmp / "pred.txt", tmp / "fold.txt"
    run(binary, table, split + ["--predictions", str(pred), "--dump-folds", str(fold), "--quiet"] + opts)
    return np.array(fold.read_text().split(), dtype=np.int64), pred.read_text().split("\n")[:-1]


class Data:
    """A table the way the C++ loader sees it: codes in order of first appearance, float32 numerics."""

    def __init__(self, table: Path) -> None:
        df = pd.read_csv(table, dtype=str, keep_default_na=False)
        types = [c.rsplit(":", 1) for c in df.columns]
        self.n = len(df)
        self.names, self.numeric, cols = [], [], []
        self.n_values: list[int] = []
        for (name, kind), col in zip(types, df.columns):
            if kind == "class":
                self.y, self.classes = pd.factorize(df[col])
                continue
            self.names.append(name)
            self.numeric.append(kind == "num")
            if kind == "num":
                cols.append(pd.to_numeric(df[col].replace("?", "nan")).to_numpy(dtype=np.float32))
                self.n_values.append(1)
            else:
                codes, values = pd.factorize(df[col])
                cols.append(codes.astype(np.int64))
                self.n_values.append(len(values))
        self.cols = cols
        self.n_classes = len(self.classes)

    def labels_to_codes(self, labels: list[str]) -> np.ndarray:
        lookup = {c: i for i, c in enumerate(self.classes)}
        return np.array([lookup.get(v, -1) for v in labels], dtype=np.int64)

    def cat_matrix(self) -> np.ndarray:
        return np.column_stack([c for c, num in zip(self.cols, self.numeric) if not num]) if not all(self.numeric) \
            else np.empty((self.n, 0), dtype=np.int64)

    def num_matrix(self) -> np.ndarray:
        return np.column_stack([c.astype(np.float64) for c, num in zip(self.cols, self.numeric) if num]) \
            if any(self.numeric) else np.empty((self.n, 0))

    def design(self, train: np.ndarray) -> np.ndarray:
        """One-hot + standardized design matrix of logistic regression (statistics from `train` rows)."""
        parts = []
        for col, num, v in zip(self.cols, self.numeric, self.n_values):
            if num:
                x = col.astype(np.float64)
                mean = x[train].mean()
                sd = x[train].std()
                parts.append((((x - mean) / (sd if sd > 0 else 1.0)).astype(np.float32).astype(np.float64))[:, None])
            else:
                parts.append(np.eye(v)[col])
        return np.hstack(parts)


# --------------------------------------------------------------- naive Bayes

def sklearn_nb(data: Data, train: np.ndarray, test: np.ndarray):
    """Predicted class code and the margin between the two best classes."""
    y = data.y[train]
    present = np.unique(y)
    total = np.zeros((int(test.sum()), len(present)))
    prior = np.log(np.bincount(y, minlength=data.n_classes)[present] / len(y))
    used = 0
    xc, xn = data.cat_matrix(), data.num_matrix()
    if xc.shape[1]:
        cat_values = [v for v, num in zip(data.n_values, data.numeric) if not num]
        m = CategoricalNB(alpha=1.0, min_categories=cat_values).fit(xc[train], y)
        total += m.predict_joint_log_proba(xc[test])
        used += 1
    if xn.shape[1]:
        m = GaussianNB(var_smoothing=1e-9).fit(xn[train], y)
        total += m.predict_joint_log_proba(xn[test])
        used += 1
    total -= (used - 1) * prior      # each model added the prior once
    order = np.sort(total, axis=1)
    margin = order[:, -1] - order[:, -2] if len(present) > 1 else np.full(len(total), np.inf)
    return present[np.argmax(total, axis=1)], margin


def test_nbayes(res: Results, bins: dict[str, Path], tmp: Path, quick: bool) -> None:
    print("\n[naive Bayes 1] Han & Kamber, Example 8.4")
    for bname, binary in bins.items():
        out = run(binary, EXAMPLES / "buys_computer.csv", ["--alpha", "0", "--query", "youth,medium,yes,fair",
                                                           "--folds", "0", "--quiet"])
        score = {m.group(1): (float(m.group(2)), float(m.group(3)))
                 for m in re.finditer(r"^\s+(\w+)\s+P\(X\|C\) (\S+)\s+P\(C\) \S+\s+P\(X\|C\)P\(C\) (\S+)", out, re.M)}
        res.check(f"P(X|yes) = 0.044, P(X|no) = 0.019 {bname}",
                  abs(score["yes"][0] - 0.044) < 1e-3 and abs(score["no"][0] - 0.019) < 1e-3, str(score))
        res.check(f"P(X|yes)P(yes) = 0.028, P(X|no)P(no) = 0.007, class = yes {bname}",
                  abs(score["yes"][1] - 0.028) < 1e-3 and abs(score["no"][1] - 0.007) < 1e-3
                  and "predicted class: yes" in out, str(score))

    print("\n[naive Bayes 2] cross-validation predictions vs scikit-learn")
    tables = random_tables(tmp, 6 if quick else 18, seed=11)
    tables += [(n, UCI / f"{n}.csv") for n in ["votes", "tictactoe", "mushroom", "nursery", "krkopt", "letter", "adult"]
               + ([] if quick else ["chess", "splice", "connect4", "covertype", "census"])]
    for name, table in tables:
        if not table.exists():
            continue
        data = Data(table)
        folds = 5 if data.n >= 50 else 2
        for bname, binary in bins.items():
            fold, labels = cpp_cv(binary, table, [], ["--folds", str(folds)], tmp)
            got = data.labels_to_codes(labels)
            wrong = ties = 0
            for f in range(folds):
                test = fold == f
                want, margin = sklearn_nb(data, ~test, test)
                differ = got[test] != want
                ties += int((differ & (margin < 1e-8)).sum())
                wrong += int((differ & (margin >= 1e-8)).sum())
            acc = float((got == data.y).mean())
            res.check(f"{name} {bname}", wrong == 0,
                      f"{data.n} records, accuracy {acc:.4f}, {wrong} differ" + (f", {ties} ties" if ties else ""),
                      show=not name.startswith("rand"))


# ------------------------------------------------------- k-nearest neighbors

def numpy_knn(data: Data, train: np.ndarray, test: np.ndarray, k: int) -> np.ndarray:
    """Exhaustive search with the rules of knn_cpu: single precision, attribute by attribute,
    every record at most as far as the k-th nearest votes, lower class id first among equal votes."""
    t_cols, q_cols = [], []
    for col, num in zip(data.cols, data.numeric):
        if num:
            x = col.astype(np.float64)
            lo, hi = x[train].min(), x[train].max()
            x = ((x - lo) / (hi - lo if hi > lo else 1.0)).astype(np.float32)
            t_cols.append(x[train])
            q_cols.append(x[test])
        else:
            t_cols.append(col[train])
            q_cols.append(col[test])
    y = data.y[train]
    n_train, n_test = int(train.sum()), int(test.sum())
    k = min(k, n_train)
    out = np.empty(n_test, dtype=np.int64)
    chunk = max(1, 8_000_000 // n_train)
    for start in range(0, n_test, chunk):
        stop = min(n_test, start + chunk)
        dist = np.zeros((stop - start, n_train), dtype=np.float32)
        for t, q, num in zip(t_cols, q_cols, data.numeric):
            if num:
                diff = t[None, :] - q[start:stop, None]
                dist += diff * diff
            else:
                dist += (t[None, :] != q[start:stop, None]).astype(np.float32)
        radius = np.partition(dist, k - 1, axis=1)[:, k - 1]
        voters = dist <= radius[:, None]
        votes = np.stack([voters[:, y == c].sum(axis=1) for c in range(data.n_classes)], axis=1)
        out[start:stop] = np.argmax(votes, axis=1)
    return out


def test_knn(res: Results, bins: dict[str, Path], tmp: Path, quick: bool) -> None:
    print("\n[k-nearest neighbors 3] cross-validation predictions vs exhaustive NumPy search")
    tables = random_tables(tmp, 6 if quick else 18, seed=23)
    runs = [(n, t, ["--folds", "3"], k) for (n, t), k in zip(tables, [1, 3, 5, 7, 4, 50] * 3)]
    runs += [("votes", UCI / "votes.csv", ["--folds", "5"], 5), ("tictactoe", UCI / "tictactoe.csv", ["--folds", "5"], 5),
             ("tictactoe", UCI / "tictactoe.csv", ["--folds", "5"], 1),
             ("mushroom", UCI / "mushroom.csv", ["--holdout", "0.2"], 5),
             ("letter", UCI / "letter.csv", ["--holdout", "0.1"], 5), ("krkopt", UCI / "krkopt.csv", ["--holdout", "0.1"], 3)]
    if not quick:
        runs += [("chess", UCI / "chess.csv", ["--folds", "3"], 5), ("nursery", UCI / "nursery.csv", ["--holdout", "0.2"], 10),
                 ("adult", UCI / "adult.csv", ["--holdout", "0.05"], 5)]
    for name, table, split, k in runs:
        if not table.exists():
            continue
        data = Data(table)
        for bname, binary in bins.items():
            fold, labels = cpp_cv(binary, table, ["--k", str(k)], split, tmp)
            got = data.labels_to_codes(labels)
            wrong = 0
            if split[0] == "--holdout":
                test = fold == 1
                wrong = int((got[test] != numpy_knn(data, ~test, test, k)).sum())
                tested = test
            else:
                for f in range(int(split[1])):
                    test = fold == f
                    wrong += int((got[test] != numpy_knn(data, ~test, test, k)).sum())
                tested = np.ones(data.n, dtype=bool)
            acc = float((got[tested] == data.y[tested]).mean())
            res.check(f"{name} k={k} {' '.join(split)} {bname}", wrong == 0,
                      f"{int(tested.sum())} tested, accuracy {acc:.4f}, {wrong} differ", show=not name.startswith("rand"))

    print("\n[k-nearest neighbors 4] scikit-learn KNeighborsClassifier on real-valued data")
    rng = random.Random(5)
    header, rows = random_table(rng, 3000, 0, 6, 4, 0.1, 2, "real", 0.0)
    table = tmp / "real.csv"
    write_table(table, header, rows)
    data = Data(table)
    x = data.num_matrix()
    for bname, binary in bins.items():
        fold, labels = cpp_cv(binary, table, ["--k", "5"], ["--folds", "5"], tmp)
        got = data.labels_to_codes(labels)
        want = np.empty(data.n, dtype=np.int64)
        for f in range(5):
            test = fold == f
            lo, hi = x[~test].min(axis=0), x[~test].max(axis=0)
            m = KNeighborsClassifier(n_neighbors=5, algorithm="brute").fit((x[~test] - lo) / (hi - lo), data.y[~test])
            want[test] = m.predict((x[test] - lo) / (hi - lo))
        agree = float((got == want).mean())
        res.check(f"real-valued table k=5 {bname}", agree >= 0.995,
                  f"agreement {agree:.4f}, accuracy ours {(got == data.y).mean():.4f}, sklearn {(want == data.y).mean():.4f}")


# ------------------------------------------------------- logistic regression

def softmax_objective(x: np.ndarray, y: np.ndarray, w: np.ndarray, b: np.ndarray, c: float):
    """Mean negative log-likelihood + ||W||^2 / (2 C n), and the class probabilities."""
    z = x @ w + b
    z -= z.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    n = len(y)
    return -logp[np.arange(n), y].mean() + (w ** 2).sum() / (2 * c * n), np.exp(logp)


def test_logreg(res: Results, bins: dict[str, Path], tmp: Path, quick: bool) -> None:
    print("\n[logistic regression 5] optimum vs scikit-learn LogisticRegression")
    tables = random_tables(tmp, 4 if quick else 10, seed=31, n_choices=[60, 200, 600])
    tables += [(n, UCI / f"{n}.csv") for n in ["votes", "tictactoe", "nursery"]
               + ([] if quick else ["mushroom", "krkopt", "adult"])]
    for i, (name, table) in enumerate(tables):
        if not table.exists():
            continue
        data = Data(table)
        c = [1.0, 0.1, 10.0][i % 3] if name.startswith("rand") else 1.0
        x = data.design(np.ones(data.n, dtype=bool))
        k = data.n_classes
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Two classes: scikit-learn fits one weight vector w = w_1 - w_0, for which
            # the softmax penalty ||w_0||^2 + ||w_1||^2 equals ||w||^2 / 2, i.e. C doubles.
            sk = LogisticRegression(C=2 * c if k == 2 else c, tol=1e-10, max_iter=20000).fit(x, data.y)
        if k == 2:
            w_sk = np.column_stack([-sk.coef_[0] / 2, sk.coef_[0] / 2])
            b_sk = np.array([-sk.intercept_[0] / 2, sk.intercept_[0] / 2])
        else:
            w_sk, b_sk = sk.coef_.T, sk.intercept_
        f_sk, p_sk = softmax_objective(x, data.y, w_sk, b_sk, c)
        for bname, binary in bins.items():
            dump = tmp / "weights.csv"
            run(binary, table, ["--C", str(c), "--tol", "1e-7", "--max-iter", "5000", "--dump-model", str(dump),
                                "--folds", "0", "--quiet"])
            w = pd.read_csv(dump, keep_default_na=False).iloc[:, 1:].to_numpy(dtype=np.float64)
            f, p = softmax_objective(x, data.y, w[:-1], w[-1], c)
            gap = float(np.abs(p - p_sk).max())
            res.check(f"{name} C={c:g} {bname}", abs(f - f_sk) < 1e-6 and gap < 2e-3,
                      f"{x.shape[1]} features x {k} classes, objective ours {f:.8f}, sklearn {f_sk:.8f}, "
                      f"largest probability gap {gap:.1e}", show=not name.startswith("rand"))

    print("\n[logistic regression 6] cross-validation predictions vs scikit-learn (default settings)")
    names = ["votes", "mushroom", "nursery"] + ([] if quick else ["letter", "adult"])
    for name in names:
        table = UCI / f"{name}.csv"
        if not table.exists():
            continue
        data = Data(table)
        k = data.n_classes
        for bname, binary in bins.items():
            fold, labels = cpp_cv(binary, table, [], ["--folds", "5"], tmp)
            got = data.labels_to_codes(labels)
            want = np.empty(data.n, dtype=np.int64)
            for f in range(5):
                test = fold == f
                x = data.design(~test)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = LogisticRegression(C=2.0 if k == 2 else 1.0).fit(x[~test], data.y[~test])
                want[test] = m.predict(x[test])
            agree, acc, sk_acc = float((got == want).mean()), float((got == data.y).mean()), float((want == data.y).mean())
            res.check(f"{name} {bname}", agree >= 0.99 and abs(acc - sk_acc) < 0.005,
                      f"agreement {agree:.4f}, accuracy ours {acc:.4f}, sklearn {sk_acc:.4f}")


def random_tables(tmp: Path, count: int, seed: int, n_choices: list[int] | None = None) -> list[tuple[str, Path]]:
    rng = random.Random(seed)
    out = []
    for i in range(count):
        n = rng.choice(n_choices or [15, 60, 200, 600])
        n_cat, n_num = [(4, 0), (0, 4), (3, 3), (6, 2), (1, 5), (2, 1)][i % 6]
        header, rows = random_table(rng, n, n_cat, n_num, rng.choice([2, 3, 5]), rng.choice([0.0, 0.1, 0.3]),
                                    rng.choice([3, 6, 14]), rng.choice(["few", "real", "many"]),
                                    rng.choice([0.0, 0.0, 0.1]))
        path = tmp / f"rand{seed}_{i:02d}.csv"
        write_table(path, header, rows)
        out.append((f"rand{i:02d}(n={n},cat={n_cat},num={n_num})", path))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin-dir", default=str(ROOT / "bin"))
    ap.add_argument("--quick", action="store_true", help="fewer random cases, skip the larger UCI runs")
    args = ap.parse_args()

    res = Results()
    tmp = Path(tempfile.mkdtemp(prefix="baseline_tests_"))
    if not UCI.exists():
        print("data/uci missing: only the examples and random tables are tested (run scripts/fetch_uci.py)")
    found = False
    for algorithm, test in (("nbayes", test_nbayes), ("knn", test_knn), ("logreg", test_logreg)):
        bins = find_binaries(Path(args.bin_dir), algorithm)
        if not bins:
            print(f"\nno {algorithm} binaries in {args.bin_dir}; skipped")
            continue
        found = True
        test(res, bins, tmp, args.quick)
    if not found:
        return 2

    print(f"\n{res.passed} passed, {len(res.failed)} failed")
    for f in res.failed:
        print("  failed:", f)
    return 1 if res.failed else 0


if __name__ == "__main__":
    sys.exit(main())
