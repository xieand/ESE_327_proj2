"""Correctness tests for the decision tree programs.

  1. Textbook examples (data/examples): attribute scores at the root against
     the numbers printed in Han & Kamber, and whole trees against the
     published ones (*.tree files).
  2. Randomized and edge-case tables: the tree and the cross-validation
     predictions must be identical to those of the independent Python
     reference (scripts/dtree_ref.py), for all three measures and for the
     depth / size / binning / pruning options.
  3. UCI benchmarks against the Python reference (run scripts/fetch_uci.py first).
  4. scikit-learn as an outside check on numeric data: same root split, and
     cross-validation accuracy and tree size close to DecisionTreeClassifier
     (not identical: scikit-learn breaks ties between equally good splits at random).

usage: python scripts/test_dtree.py [--bin-dir bin] [--quick]
Every binary found among dtree_cpu, dtree_gpu and dtree_gpu_emu is tested.
"""
from __future__ import annotations

import argparse
import os
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from dtree_ref import ReferenceTree, Table

ROOT = Path(__file__).resolve().parent.parent
MEASURES = ["gain", "gain_ratio", "gini"]


def find_binaries(bin_dir: Path) -> dict[str, Path]:
    exe = ".exe" if os.name == "nt" else ""
    return {n: bin_dir / f"{n}{exe}" for n in ("dtree_cpu", "dtree_gpu", "dtree_gpu_emu")
            if (bin_dir / f"{n}{exe}").exists()}


def run(binary: Path, table: Path, args: list[str]) -> str:
    cmd = [str(binary), str(table)] + args
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed ({proc.returncode}):\n{proc.stderr}")
    return proc.stdout


def cpp_tree(binary: Path, table: Path, measure: str, opts: list[str]) -> list[str]:
    out = run(binary, table, ["--measure", measure, "--print-tree", "--folds", "0", "--quiet"] + opts)
    return out.splitlines()


def cpp_scores(binary: Path, table: Path, measure: str) -> dict[str, dict[str, float]]:
    out = run(binary, table, ["--measure", measure, "--scores", "--folds", "0", "--quiet"])
    scores = {}
    for line in out.splitlines()[1:]:
        name = line.split()[0]
        scores[name] = {k: float(v) for k, v in re.findall(r"(\w+) (-?\d+\.\d+)", line)}
        scores[name]["split"] = line.split("  ")[-1].strip()
    return scores


def cpp_cv(binary: Path, table: Path, measure: str, opts: list[str], folds: int, tmp: Path):
    pred, fold = tmp / "pred.txt", tmp / "fold.txt"
    run(binary, table, ["--measure", measure, "--folds", str(folds), "--predictions", str(pred), "--dump-folds",
                        str(fold), "--quiet"] + opts)
    return [int(x) for x in fold.read_text().split()], pred.read_text().split("\n")[:-1]


def ref_kwargs(opts: list[str]) -> dict:
    """Translate dtree_cpu options into ReferenceTree arguments."""
    kw: dict = {}
    i = 0
    while i < len(opts):
        if opts[i] == "--prune":
            kw["prune_cf"] = 0.25
            i += 1
            continue
        key = {"--max-depth": "max_depth", "--min-split": "min_split", "--max-bins": "max_bins",
               "--prune-cf": "prune_cf"}[opts[i]]
        kw[key] = float(opts[i + 1]) if key == "prune_cf" else int(opts[i + 1])
        i += 2
    return kw


class Results:
    def __init__(self) -> None:
        self.passed = 0
        self.failed: list[str] = []

    def check(self, name: str, ok: bool, detail: str = "", show: bool = True) -> None:
        if ok:
            self.passed += 1
            if show:
                print(f"  ok   {name}" + (f"  ({detail})" if detail else ""))
        else:
            self.failed.append(name)
            print(f"  FAIL {name}  {detail}")

    def same_tree(self, name: str, got: list[str], want: list[str], show: bool = True) -> None:
        detail = f"{len(want)} nodes"
        if got != want:
            k = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
            detail = (f"first difference at line {k + 1}:\n         got : {got[k] if k < len(got) else '<end>'}"
                      f"\n         want: {want[k] if k < len(want) else '<end>'}")
        self.check(name, got == want, detail, show)


# ---------------------------------------------------------------- test tables

def write_table(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.write_text(",".join(header) + "\n" + "".join(",".join(r) + "\n" for r in rows), encoding="utf-8")


def random_table(rng: random.Random, n: int, n_cat: int, n_num: int, n_classes: int, noise: float,
                 cat_values: int, num_kind: str, missing: float) -> tuple[list[str], list[list[str]]]:
    """Records whose class depends on a few attributes, plus label noise."""
    header, cols = [], []
    for j in range(n_cat):
        v = rng.randint(2, cat_values)
        header.append(f"c{j}:cat")
        cols.append([("?" if rng.random() < missing else f"v{rng.randrange(v)}") for _ in range(n)])
    for j in range(n_num):
        header.append(f"x{j}:num")
        if num_kind == "few":
            cols.append([str(rng.randint(0, 7)) for _ in range(n)])
        elif num_kind == "real":
            cols.append([f"{rng.gauss(0, 1):.4f}" for _ in range(n)])
        else:   # many distinct integers, some negative
            cols.append([str(rng.randint(-400, 400)) for _ in range(n)])
    d = n_cat + n_num
    used = rng.sample(range(d), min(d, 3))
    weights = [rng.randint(1, 5) for _ in used]
    labels = []
    for i in range(n):
        s = 0
        for a, w in zip(used, weights):
            v = cols[a][i]
            s += w * (hash_value(v) if a < n_cat else int(float(v) > 0) + int(float(v) > 3))
        c = s % n_classes
        if rng.random() < noise:
            c = rng.randrange(n_classes)
        labels.append(f"k{c}")
    header.append("label:class")
    return header, [[col[i] for col in cols] + [labels[i]] for i in range(n)]


def hash_value(v: str) -> int:
    return sum(map(ord, v)) % 7


def edge_cases() -> list[tuple[str, list[str], list[list[str]]]]:
    xor = [[a, b, str(int(a != b))] for a in "01" for b in "01"] * 3
    many = [[f"v{i % 70}", str(i % 5), f"k{(i % 70) % 3}"] for i in range(420)]
    return [
        ("single-record", ["a:cat", "x:num", "y:class"], [["p", "1", "yes"]]),
        ("single-class", ["a:cat", "x:num", "y:class"], [["p", "1", "yes"], ["q", "2", "yes"], ["p", "3", "yes"]]),
        ("identical-records-two-classes", ["a:cat", "x:num", "y:class"], [["p", "1", "yes"], ["p", "1", "no"]] * 3),
        ("constant-attribute", ["a:cat", "b:cat", "y:class"], [["p", "u", "yes"], ["p", "v", "no"], ["p", "u", "yes"]]),
        ("xor-zero-gain-root", ["a:cat", "b:cat", "y:class"], xor),
        ("xor-numeric", ["a:num", "b:num", "y:class"], xor),
        ("missing-numeric", ["x:num", "y:class"], [["1", "a"], ["?", "b"], ["2", "a"], ["?", "b"], ["3", "b"]]),
        ("two-records", ["x:num", "y:class"], [["1.5", "a"], ["2.5", "b"]]),
        ("70-value-attribute", ["a:cat", "x:num", "y:class"], many),
        ("class-tie", ["a:cat", "y:class"], [["p", "no"], ["p", "yes"], ["q", "yes"], ["q", "no"]]),
    ]


OPTION_SETS = [[], ["--max-depth", "3"], ["--min-split", "8"], ["--prune"], ["--max-bins", "5"],
               ["--prune-cf", "0.1", "--max-depth", "6"]]


def compare_with_reference(res: Results, bins: dict[str, Path], table: Path, name: str, opts: list[str],
                           tmp: Path, cv_folds: int = 0, show: bool = False) -> None:
    t = Table(str(table))
    for measure in MEASURES:
        ref = ReferenceTree(t, measure, **ref_kwargs(opts)).fit()
        for bname, binary in bins.items():
            label = f"{name} {measure} {' '.join(opts)} {bname}".replace("  ", " ")
            res.same_tree(f"{label} tree", cpp_tree(binary, table, measure, opts), ref.lines(), show)
        if cv_folds:
            for bname, binary in bins.items():
                fold, pred = cpp_cv(binary, table, measure, opts, cv_folds, tmp)
                want = [""] * len(fold)
                for f in range(cv_folds):
                    fit = ReferenceTree(t, measure, **ref_kwargs(opts)).fit([i for i, v in enumerate(fold) if v != f])
                    for i, v in enumerate(fold):
                        if v == f:
                            want[i] = t.classes[fit.predict_row(i)]
                wrong = sum(a != b for a, b in zip(pred, want))
                res.check(f"{name} {measure} {' '.join(opts)} {bname} {cv_folds}-fold predictions",
                          wrong == 0 and len(pred) == len(want), f"{wrong} of {len(want)} differ", show)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin-dir", default=str(ROOT / "bin"))
    ap.add_argument("--quick", action="store_true", help="fewer random cases, skip the larger UCI runs")
    args = ap.parse_args()

    bins = find_binaries(Path(args.bin_dir))
    if not bins:
        print(f"no binaries in {args.bin_dir}; build first (make / cmake)")
        return 2
    print("testing:", ", ".join(bins))
    res = Results()
    tmp = Path(tempfile.mkdtemp(prefix="dtree_tests_"))
    examples = ROOT / "data" / "examples"

    # ------------------------------------------------------------------ 1
    print("\n[1] textbook examples")
    book_gain = {"age": 0.246, "income": 0.029, "student": 0.151, "credit_rating": 0.048}   # Han & Kamber, Ex. 8.1
    book_gini = {"age": (0.357, "in {youth, senior}"), "income": (0.443, "in {high}"),      # Ex. 8.3
                 "student": (0.367, None), "credit_rating": (0.429, None)}
    for bname, binary in bins.items():
        s = cpp_scores(binary, examples / "buys_computer.csv", "gain")
        res.check(f"buys_computer Gain(A) = {book_gain} {bname}",
                  all(abs(s[a]["gain"] - g) < 1e-3 for a, g in book_gain.items()))
        res.check(f"buys_computer GainRatio(income) = 0.019, SplitInfo = 1.557 {bname}",
                  abs(s["income"]["gain_ratio"] - 0.019) < 1e-3 and abs(s["income"]["split_info"] - 1.557) < 1e-3)
        s = cpp_scores(binary, examples / "buys_computer.csv", "gini")
        res.check(f"buys_computer Gini splits = {{age: 0.357, income: 0.443, student: 0.367, credit_rating: 0.429}} {bname}",
                  all(abs(s[a]["gini_split"] - g) < 1e-3 and (sp is None or s[a]["split"] == sp)
                      for a, (g, sp) in book_gini.items()))
        res.check(f"buys_computer Gini(D) = 0.459 {bname}",
                  abs(s["age"]["gini_split"] + s["age"]["reduction"] - 0.459) < 1e-3)
        for expected in sorted(examples.glob("*.tree")):
            stem, measure = expected.name.rsplit(".", 2)[:2]
            res.same_tree(f"{stem} {measure} {bname} published tree",
                          cpp_tree(binary, examples / f"{stem}.csv", measure, []), expected.read_text().splitlines())
    for table in sorted(examples.glob("*.csv")):
        compare_with_reference(res, bins, table, table.stem, [], tmp, show=True)

    # ------------------------------------------------------------------ 2
    print("\n[2] edge cases and randomized tables vs the Python reference")
    before = res.passed
    table = tmp / "case.csv"
    for name, header, rows in edge_cases():
        write_table(table, header, rows)
        for opts in ([], ["--prune"]):
            compare_with_reference(res, bins, table, name, opts, tmp)
    rng = random.Random(327)
    n_random = 18 if args.quick else 72
    for i in range(n_random):
        n = rng.choice([12, 40, 120, 400])
        n_cat, n_num = rng.choice([(4, 0), (0, 4), (3, 3), (6, 2), (1, 5)])
        header, rows = random_table(rng, n, n_cat, n_num, rng.choice([2, 2, 3, 5]), rng.choice([0.0, 0.1, 0.3]),
                                    rng.choice([3, 6, 14]), rng.choice(["few", "real", "many"]),
                                    rng.choice([0.0, 0.0, 0.1]))
        write_table(table, header, rows)
        opts = OPTION_SETS[i % len(OPTION_SETS)]
        compare_with_reference(res, bins, table, f"rand{i:02d}(n={n},cat={n_cat},num={n_num})", opts, tmp,
                               cv_folds=3 if i % 3 == 0 and n >= 40 else 0)
    print(f"  {res.passed - before} checks passed")

    # ------------------------------------------------------------------ 3
    print("\n[3] UCI benchmarks vs the Python reference")
    uci = ROOT / "data" / "uci"
    runs = [("tictactoe", []), ("votes", []), ("votes", ["--prune"]), ("mushroom", []), ("chess", []),
            ("nursery", ["--prune"])]
    if not args.quick:
        runs += [("splice", []), ("letter", ["--max-depth", "5"]), ("krkopt", ["--max-depth", "4"]),
                 ("adult", ["--max-depth", "4"]), ("adult", ["--max-depth", "4", "--max-bins", "16"])]
    if not uci.exists():
        print("  skipped (run: python scripts/fetch_uci.py)")
    else:
        for ds, opts in runs:
            if (uci / f"{ds}.csv").exists():
                compare_with_reference(res, bins, uci / f"{ds}.csv", ds, opts, tmp, show=True)

    # ------------------------------------------------------------------ 4
    print("\n[4] scikit-learn DecisionTreeClassifier on numeric data")
    try:
        import numpy as np
        from sklearn.tree import DecisionTreeClassifier
    except ImportError:
        print("  skipped (pip install scikit-learn)")
    else:
        tables = []
        rng = random.Random(7)
        header, rows = random_table(rng, 2000, 0, 6, 3, 0.1, 2, "real", 0.0)
        write_table(tmp / "real.csv", header, rows)
        tables.append(tmp / "real.csv")
        if (uci / "letter.csv").exists():
            tables.append(uci / "letter.csv")
        for table in tables:
            raw = np.loadtxt(table, delimiter=",", skiprows=1, dtype=str)
            x, y = raw[:, :-1].astype(np.float32), raw[:, -1]
            names = [h.split(":")[0] for h in table.read_text().splitlines()[0].split(",")]
            for measure, criterion in (("gain", "entropy"), ("gini", "gini")):
                sk = DecisionTreeClassifier(criterion=criterion, random_state=0).fit(x, y)
                for bname, binary in bins.items():
                    exact = ["--max-bins", "0"]
                    root = cpp_tree(binary, table, measure, exact + ["--max-depth", "1"])
                    want_attr, want_thr = names[sk.tree_.feature[0]], float(sk.tree_.threshold[0])
                    got_attr = root[0].split("]")[0][1:]
                    got_thr = float(root[1].split()[1].rstrip(":"))
                    res.check(f"{table.stem} {measure} {bname} root split = sklearn ({want_attr} <= {want_thr:g})",
                              got_attr == want_attr and abs(got_thr - want_thr) < 1e-6, f"got {got_attr} <= {got_thr:g}")
                    fold, pred = cpp_cv(binary, table, measure, exact, 10, tmp)
                    fold = np.array(fold)
                    sk_pred = np.empty(len(y), dtype=object)
                    for f in range(10):
                        m = DecisionTreeClassifier(criterion=criterion, random_state=0).fit(x[fold != f], y[fold != f])
                        sk_pred[fold == f] = m.predict(x[fold == f])
                    acc, sk_acc = float((np.array(pred) == y).mean()), float((sk_pred == y).mean())
                    res.check(f"{table.stem} {measure} {bname} 10-fold accuracy close to sklearn",
                              abs(acc - sk_acc) < 0.015, f"ours {acc:.4f}, sklearn {sk_acc:.4f}")
                    full = cpp_tree(binary, table, measure, exact)
                    leaves = sum("leaf " in ln for ln in full)
                    res.check(f"{table.stem} {measure} {bname} tree size close to sklearn",
                              abs(leaves - sk.get_n_leaves()) <= 0.03 * sk.get_n_leaves(),
                              f"ours {leaves} leaves, sklearn {sk.get_n_leaves()}")

    print(f"\n{res.passed} passed, {len(res.failed)} failed")
    for f in res.failed:
        print("  failed:", f)
    return 1 if res.failed else 0


if __name__ == "__main__":
    sys.exit(main())
