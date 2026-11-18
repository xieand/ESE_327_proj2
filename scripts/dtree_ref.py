"""Independent reference implementation of the decision tree (pure Python).

It follows the textbook recursion (Han & Kamber, Generate_decision_tree)
literally: every node recounts its own records and scores every candidate
split with the defining formulas (entropy, Info_A, SplitInfo, Gini). It shares
no code or data structures with the C++ programs, so equal trees mean the
C++ histograms, in-place partitioning and closed-form scores are right.

The rules that make the tree unique are the ones documented in
include/dtree.h: candidate order, the 1e-9 tie tolerance, the gain-ratio
average-gain rule, the subset search of the Gini index, empty branches, and
pessimistic pruning.

usage: python scripts/dtree_ref.py <table.csv> [--measure gain|gain_ratio|gini] [--max-depth D]
                                   [--min-split N] [--max-bins B] [--prune-cf CF]
       prints the tree trained on the whole table, in the format of dtree_cpu --print-tree
"""
from __future__ import annotations

import argparse
import math
import struct
import sys

EPS = 1e-9
PRUNE_SLACK = 0.1
EXHAUSTIVE_VALUES = 10
MAX_BIN_ID = 65535
INF = float("inf")

sys.setrecursionlimit(100000)


def f32(x: float) -> float:
    """Round to single precision, the storage type of the C++ table."""
    return struct.unpack("f", struct.pack("f", x))[0]


class Table:
    def __init__(self, path: str) -> None:
        with open(path, encoding="utf-8") as f:
            lines = [ln.rstrip("\r\n") for ln in f if ln.strip()]
        header = [h.strip().rsplit(":", 1) for h in lines[0].split(",")]
        class_col = next(i for i, (_, t) in enumerate(header) if t == "class")
        self.class_name = header[class_col][0]
        self.names = [n for n, t in header if t != "class"]
        self.numeric = [t == "num" for _, t in header if t != "class"]
        self.values: list[list[str]] = [[] for _ in self.names]      # categorical labels by code
        self.classes: list[str] = []
        self.x: list[list[float]] = []                                # x[row][attr]: value or code
        self.y: list[int] = []
        codes: list[dict[str, int]] = [{} for _ in self.names]
        class_codes: dict[str, int] = {}
        for ln in lines[1:]:
            fields = [v.strip() for v in ln.split(",")]
            row = []
            a = 0
            for i, v in enumerate(fields):
                if i == class_col:
                    if v not in class_codes:
                        class_codes[v] = len(self.classes)
                        self.classes.append(v)
                    self.y.append(class_codes[v])
                    continue
                if self.numeric[a]:
                    row.append(INF if v in ("", "?") else f32(float(v)))
                else:
                    v = v or "?"
                    if v not in codes[a]:
                        codes[a][v] = len(self.values[a])
                        self.values[a].append(v)
                    row.append(codes[a][v])
                a += 1
            self.x.append(row)
        self.n_classes = len(self.classes)


# ------------------------------------------------------------------ measures

def entropy(counts: list[int]) -> float:
    n = sum(counts)
    return -sum(c / n * math.log2(c / n) for c in counts if c)


def gini(counts: list[int]) -> float:
    n = sum(counts)
    return 1.0 - sum((c / n) ** 2 for c in counts)


def weighted(parts: list[list[int]], impurity) -> float:
    n = sum(sum(p) for p in parts)
    return sum(sum(p) / n * impurity(p) for p in parts)


def split_info(parts: list[list[int]]) -> float:
    return entropy([sum(p) for p in parts])


def add(a: list[int], b: list[int]) -> list[int]:
    return [x + y for x, y in zip(a, b)]


def sub(a: list[int], b: list[int]) -> list[int]:
    return [x - y for x, y in zip(a, b)]


def majority(counts: list[int]) -> int:
    return max(range(len(counts)), key=lambda c: (counts[c], -c))


# ---------------------------------------------------------------------- tree

class Node:
    def __init__(self, counts: list[int]) -> None:
        self.counts = counts
        self.n = sum(counts)
        self.label = majority(counts)
        self.kind = "leaf"
        self.attr = -1
        self.threshold = 0.0
        self.left_values: list[int] = []
        self.children: list[Node] = []


class ReferenceTree:
    def __init__(self, table: Table, measure: str = "gain", max_depth: int = 0, min_split: int = 2,
                 max_bins: int = 256, prune_cf: float = 0.0) -> None:
        self.t = table
        self.measure = measure
        self.max_depth = max_depth
        self.min_split = min_split
        self.max_bins = max_bins
        self.prune_cf = prune_cf
        self.root: Node | None = None

    # ---- binning of numeric attributes (one bin per value unless there are too many)

    def _fit_bins(self, rows: list[int]) -> None:
        t = self.t
        limit = MAX_BIN_ID if self.max_bins <= 0 else min(self.max_bins, MAX_BIN_ID)
        self.bin_of: list[dict[float, int]] = []
        self.lo: list[list[float]] = []
        self.hi: list[list[float]] = []
        n = len(rows)
        for a in range(len(t.names)):
            if not t.numeric[a]:
                self.bin_of.append({})
                self.lo.append([])
                self.hi.append([])
                continue
            freq: dict[float, int] = {}
            for r in rows:
                freq[t.x[r][a]] = freq.get(t.x[r][a], 0) + 1
            distinct = sorted(freq)
            merge = len(distinct) > limit
            bin_of, lo, hi = {}, [], []
            closed, seen = 0, 0
            for v in distinct:
                if len(lo) == closed:
                    lo.append(v)
                    hi.append(v)
                else:
                    hi[-1] = v
                bin_of[v] = len(lo) - 1
                seen += freq[v]
                if not merge or seen * limit >= (closed + 1) * n:
                    closed += 1
            self.bin_of.append(bin_of)
            self.lo.append(lo)
            self.hi.append(hi)

    def _bin(self, row: int, a: int) -> int:
        v = self.t.x[row][a]
        return self.bin_of[a][v] if self.t.numeric[a] else v

    def _threshold(self, a: int, left: int, right: int) -> float:
        below, above = self.hi[a][left], self.lo[a][right]
        mid = f32(0.5 * (below + above))
        return mid if mid < above else below

    # ---- induction

    def fit(self, rows: list[int] | None = None) -> "ReferenceTree":
        rows = list(range(len(self.t.y))) if rows is None else list(rows)
        self._fit_bins(rows)
        self.root = self._grow(rows, 0, None)
        if self.prune_cf > 0:
            self._prune(self.root)
        return self

    def _class_counts(self, rows: list[int]) -> list[int]:
        counts = [0] * self.t.n_classes
        for r in rows:
            counts[self.t.y[r]] += 1
        return counts

    def _grow(self, rows: list[int], depth: int, parent_label: int | None) -> Node:
        node = Node(self._class_counts(rows))
        if not rows:
            node.label = parent_label          # empty branch: majority class of the parent
            return node
        if max(node.counts) == node.n or node.n < self.min_split or (self.max_depth > 0 and depth >= self.max_depth):
            return node
        best = self._choose(rows, node.counts)
        if best is None:
            return node
        a, cand = best
        node.attr = a
        node.kind = cand["kind"]
        if node.kind == "multiway":
            groups = [[] for _ in self.t.values[a]]
            for r in rows:
                groups[self.t.x[r][a]].append(r)
        elif node.kind == "threshold":
            node.threshold = self._threshold(a, cand["left_bin"], cand["right_bin"])
            groups = [[r for r in rows if self._bin(r, a) <= cand["left_bin"]],
                      [r for r in rows if self._bin(r, a) > cand["left_bin"]]]
        else:
            node.left_values = cand["left_values"]
            left = set(node.left_values)
            groups = [[r for r in rows if self.t.x[r][a] in left], [r for r in rows if self.t.x[r][a] not in left]]
        node.children = [self._grow(g, depth + 1, node.label) for g in groups]
        return node

    def _choose(self, rows: list[int], counts: list[int]):
        cands = [self._score(a, rows, counts) for a in range(len(self.t.names))]
        best = None
        if self.measure == "gain_ratio":
            valid = [c for c in cands if c]
            average = sum(c["gain"] for c in valid) / len(valid) if valid else 0.0
            for a, c in enumerate(cands):
                if not c or c["gain"] < average - EPS:
                    continue
                if best is None or c["gain_ratio"] > cands[best]["gain_ratio"] + EPS:
                    best = a
        else:
            key = "gain" if self.measure == "gain" else "reduction"
            for a, c in enumerate(cands):
                if c and (best is None or c[key] > cands[best][key] + EPS):
                    best = a
        return None if best is None else (best, cands[best])

    def _score(self, a: int, rows: list[int], counts: list[int]):
        """Best split of attribute a, or None if all records share one value."""
        hist: dict[int, list[int]] = {}
        for r in rows:
            hist.setdefault(self._bin(r, a), [0] * self.t.n_classes)[self.t.y[r]] += 1
        present = sorted(hist)
        if len(present) < 2:
            return None
        numeric = self.t.numeric[a]
        if self.measure != "gini":
            if not numeric:
                parts = [hist[b] for b in present]
                g = entropy(counts) - weighted(parts, entropy)
                return {"kind": "multiway", "gain": g, "gain_ratio": g / split_info(parts)}
            best = None
            left = [0] * self.t.n_classes
            for k in range(len(present) - 1):
                left = add(left, hist[present[k]])
                parts = [left, sub(counts, left)]
                g = entropy(counts) - weighted(parts, entropy)
                if best is None or g > best["gain"] + EPS:
                    best = {"kind": "threshold", "gain": g, "gain_ratio": g / split_info(parts),
                            "left_bin": present[k], "right_bin": present[k + 1]}
            return best

        if numeric:
            best = None
            left = [0] * self.t.n_classes
            for k in range(len(present) - 1):
                left = add(left, hist[present[k]])
                red = gini(counts) - weighted([left, sub(counts, left)], gini)
                if best is None or red > best["reduction"] + EPS:
                    best = {"kind": "threshold", "reduction": red, "left_bin": present[k],
                            "right_bin": present[k + 1]}
            return best

        # categorical attribute, Gini index: binary subset split
        t = len(present)
        candidates: list[list[int]] = []       # each: the present values of the left child
        if sum(1 for c in counts if c) > 2 and t <= EXHAUSTIVE_VALUES:
            for s in range(1, 1 << (t - 1)):
                candidates.append([present[k] for k in range(t - 1) if s >> k & 1])
        else:
            m = majority(counts)
            # ascending fraction of the majority class, compared exactly
            import functools

            def cmp(x: int, y: int) -> int:
                lhs = hist[x][m] * sum(hist[y])
                rhs = hist[y][m] * sum(hist[x])
                return (lhs > rhs) - (lhs < rhs) or (x > y) - (x < y)

            order = sorted(present, key=functools.cmp_to_key(cmp))
            candidates = [order[:k] for k in range(1, t)]
        best = None
        for values in candidates:
            left = [0] * self.t.n_classes
            for v in values:
                left = add(left, hist[v])
            red = gini(counts) - weighted([left, sub(counts, left)], gini)
            if best is None or red > best["reduction"] + EPS:
                best = {"kind": "subset", "reduction": red, "present_left": values, "n_left": sum(left)}
        left_values = set(best["present_left"])
        if 2 * best["n_left"] >= len(rows):        # values the node does not contain follow the larger child
            left_values |= set(range(len(self.t.values[a]))) - set(present)
        best["left_values"] = sorted(left_values)
        return best

    # ---- pessimistic pruning

    def _pessimistic(self, n: int, e: int) -> float:
        """Quinlan's estimate (C4.5, AddErrs): e plus the extra errors expected on unseen data."""
        cf = self.prune_cf
        if n == 0:
            return 0.0
        if e == 0:
            return n * (1 - math.exp(math.log(cf) / n))
        if e + 0.5 >= n:
            return e + 0.67 * (n - e)
        val = [0, 0.001, 0.005, 0.01, 0.05, 0.10, 0.20, 0.40, 1.00]
        dev = [4.0, 3.09, 2.58, 2.33, 1.65, 1.28, 0.84, 0.25, 0.00]
        i = next(k for k in range(1, len(val)) if cf <= val[k])
        coeff = (dev[i - 1] + (dev[i] - dev[i - 1]) * (cf - val[i - 1]) / (val[i] - val[i - 1])) ** 2
        pr = (e + 0.5 + coeff / 2 + math.sqrt(coeff * ((e + 0.5) * (1 - (e + 0.5) / n) + coeff / 4))) / (n + coeff)
        return n * pr

    def _prune(self, node: Node) -> float:
        as_leaf = self._pessimistic(node.n, node.n - node.counts[node.label])
        if node.kind == "leaf":
            return as_leaf
        subtree = sum(self._prune(c) for c in node.children)
        if as_leaf <= subtree + PRUNE_SLACK:
            node.kind = "leaf"
            node.children = []
            return as_leaf
        return subtree

    # ---- use

    def predict_row(self, row: int) -> int:
        node = self.root
        while node.kind != "leaf":
            v = self.t.x[row][node.attr]
            if node.kind == "multiway":
                node = node.children[v]
            elif node.kind == "threshold":
                node = node.children[0 if v <= node.threshold else 1]
            else:
                node = node.children[0 if v in node.left_values else 1]
        return node.label

    def stats(self) -> tuple[int, int, int]:
        """(nodes, leaves, depth)"""
        def walk(node: Node, depth: int) -> tuple[int, int, int]:
            if node.kind == "leaf":
                return 1, 1, depth
            sub_stats = [walk(c, depth + 1) for c in node.children]
            return (1 + sum(s[0] for s in sub_stats), sum(s[1] for s in sub_stats), max(s[2] for s in sub_stats))
        return walk(self.root, 0)

    def lines(self) -> list[str]:
        out: list[str] = []
        t = self.t

        def walk(node: Node, edge: str, indent: int) -> None:
            head = f"leaf {t.classes[node.label]}" if node.kind == "leaf" else f"[{t.names[node.attr]}]"
            counts = ", ".join(f"{t.classes[c]}:{k}" for c, k in enumerate(node.counts))
            out.append(f"{'  ' * indent}{edge}{head} n={node.n} {{{counts}}}")
            if node.kind == "multiway":
                for v, child in enumerate(node.children):
                    walk(child, f"= {t.values[node.attr][v]}: ", indent + 1)
            elif node.kind == "threshold":
                thr = "%.9g" % node.threshold
                walk(node.children[0], f"<= {thr}: ", indent + 1)
                walk(node.children[1], f"> {thr}: ", indent + 1)
            elif node.kind == "subset":
                names = ", ".join(t.values[node.attr][v] for v in node.left_values)
                walk(node.children[0], f"in {{{names}}}: ", indent + 1)
                walk(node.children[1], f"not in {{{names}}}: ", indent + 1)

        walk(self.root, "", 0)
        return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("table")
    ap.add_argument("--measure", default="gain", choices=["gain", "gain_ratio", "gini"])
    ap.add_argument("--max-depth", type=int, default=0)
    ap.add_argument("--min-split", type=int, default=2)
    ap.add_argument("--max-bins", type=int, default=256)
    ap.add_argument("--prune-cf", type=float, default=0.0)
    args = ap.parse_args()
    tree = ReferenceTree(Table(args.table), args.measure, args.max_depth, args.min_split, args.max_bins,
                         args.prune_cf).fit()
    print("\n".join(tree.lines()))


if __name__ == "__main__":
    main()
