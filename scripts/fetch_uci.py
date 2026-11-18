"""Download the UCI benchmarks (the same 13 datasets as Project 1) and convert
them to classification tables.

Output: data/uci/<name>.csv, one record per line. The header names every
column and its type as "name:type":

    num    numeric attribute
    cat    categorical attribute
    class  the class label (always the last column)

Missing values are written as "?". Unlike the Project 1 transaction files,
numeric attributes are kept numeric (no binning) and every record keeps all of
its columns.

Raw downloads are cached in data/raw/. If a sibling ../ESE_327_proj1/data/raw
exists, its files are reused instead of downloading again.

usage: python scripts/fetch_uci.py [names...]     (default: all)
       names: mushroom chess connect4 adult tictactoe votes splice nursery
              letter krkopt poker covertype census
"""
from __future__ import annotations

import gzip
import io
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "uci"
PROJ1_RAW = ROOT.parent / "ESE_327_proj1" / "data" / "raw"
BASE = "https://archive.ics.uci.edu/static/public"

MUSHROOM_ATTRS = [
    "class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor", "gill-attachment", "gill-spacing",
    "gill-size", "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring",
    "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color",
    "ring-number", "ring-type", "spore-print-color", "population", "habitat",
]
CHESS_ATTRS = [
    "bkblk", "bknwy", "bkon8", "bkona", "bkspr", "bkxbq", "bkxcr", "bkxwp", "blxwp", "bxqsq", "cntxt", "dsopp",
    "dwipd", "hdchk", "katri", "mulch", "qxmsq", "r2ar8", "reskd", "reskr", "rimmx", "rkxwp", "rxmsq", "simpl",
    "skach", "skewr", "skrxp", "spcop", "stlmt", "thrsk", "wkcti", "wkna8", "wknck", "wkovl", "wkpos", "wtoeg",
    "class",
]
CONNECT4_ATTRS = [f"{c}{r}" for c in "abcdefg" for r in range(1, 7)] + ["class"]
TICTACTOE_ATTRS = ["tl", "tm", "tr", "ml", "mm", "mr", "bl", "bm", "br", "class"]
VOTES_ATTRS = [
    "party", "handicapped-infants", "water-project-cost-sharing", "adoption-of-the-budget-resolution",
    "physician-fee-freeze", "el-salvador-aid", "religious-groups-in-schools", "anti-satellite-test-ban",
    "aid-to-nicaraguan-contras", "mx-missile", "immigration", "synfuels-corporation-cutback", "education-spending",
    "superfund-right-to-sue", "crime", "duty-free-exports", "export-administration-act-south-africa",
]
NURSERY_ATTRS = ["parents", "has_nurs", "form", "children", "housing", "finance", "social", "health", "class"]
LETTER_ATTRS = [
    "letter", "x-box", "y-box", "width", "high", "onpix", "x-bar", "y-bar", "x2bar", "y2bar", "xybar", "x2ybr",
    "xy2br", "x-ege", "xegvy", "y-ege", "yegvx",
]
KRKOPT_ATTRS = ["wk-file", "wk-rank", "wr-file", "wr-rank", "bk-file", "bk-rank", "result"]
ADULT_COLS = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship",
    "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income",
]
ADULT_NUMERIC = {"age", "education-num", "capital-gain", "capital-loss", "hours-per-week"}
COVTYPE_NUMERIC = [
    "elevation", "aspect", "slope", "hdist-hydro", "vdist-hydro", "hdist-road", "hillshade-9am", "hillshade-noon",
    "hillshade-3pm", "hdist-fire",
]
# census-income.names order.
CENSUS_COLS = [
    "age", "class-of-worker", "industry-code", "occupation-code", "education", "wage-per-hour", "enrolled-in-edu",
    "marital-status", "major-industry", "major-occupation", "race", "hispanic-origin", "sex", "labor-union",
    "unemployment-reason", "employment-status", "capital-gains", "capital-losses", "dividends", "tax-filer-status",
    "prev-region", "prev-state", "household-stat", "household-summary", "instance-weight", "migration-msa",
    "migration-reg", "migration-within-reg", "same-house-1yr", "migration-sunbelt", "num-persons-worked-for-employer",
    "family-under-18", "father-birth-country", "mother-birth-country", "birth-country", "citizenship",
    "own-business", "veterans-questionnaire", "veterans-benefits", "weeks-worked", "year", "income",
]
CENSUS_NUMERIC = {
    "age", "wage-per-hour", "capital-gains", "capital-losses", "dividends", "num-persons-worked-for-employer",
    "weeks-worked",
}


def download(url: str, name: str) -> bytes:
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / name
    if not path.exists():
        cached = PROJ1_RAW / name
        if cached.exists():
            print(f"  reusing {cached}")
            shutil.copyfile(cached, path)
        else:
            print(f"  downloading {url}")
            with urllib.request.urlopen(url) as r:
                path.write_bytes(r.read())
    return path.read_bytes()


def zip_member(archive: bytes, member: str) -> bytes:
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        return z.read(member)


def unlzw(data: bytes) -> bytes:
    """Decompress Unix 'compress' (.Z) data."""
    if data[:2] != b"\x1f\x9d":
        raise ValueError("not a .Z file")
    max_bits = data[2] & 0x1F
    block_mode = bool(data[2] & 0x80)
    first = 257 if block_mode else 256
    table = [bytes([i]) for i in range(256)] + ([b""] if block_mode else [])
    out = bytearray()
    total_bits = len(data) * 8
    bits = 9
    seg_start = 24      # codes are written in groups of 8; a width change or
    n_in_seg = 0        # CLEAR pads to the end of the current group
    prev = None
    while True:
        if len(table) > (1 << bits) - 1 and bits < max_bits and prev is not None:
            seg_start += ((n_in_seg + 7) // 8) * 8 * bits
            n_in_seg = 0
            bits += 1
        pos = seg_start + n_in_seg * bits
        if pos + bits > total_bits:
            break
        byte = pos >> 3
        chunk = int.from_bytes(data[byte:byte + 4], "little")
        code = (chunk >> (pos & 7)) & ((1 << bits) - 1)
        n_in_seg += 1
        if block_mode and code == 256:
            seg_start += ((n_in_seg + 7) // 8) * 8 * bits
            n_in_seg = 0
            bits = 9
            del table[first:]
            prev = None
            continue
        if code < len(table):
            entry = table[code]
        elif code == len(table) and prev is not None:
            entry = prev + prev[:1]
        else:
            raise ValueError("corrupt .Z stream")
        out += entry
        if prev is not None and len(table) < (1 << max_bits):
            table.append(prev + entry[:1])
        prev = entry
    return bytes(out)


def clean(v: str) -> str:
    """Values must not contain separators; spaces become underscores."""
    return v.strip().replace(",", ";").replace(" ", "_")


def write(name: str, columns: list[tuple[str, str]], rows: list[list[str]]) -> None:
    """columns: (name, type) with exactly one 'class' column, which must be last."""
    assert [t for _, t in columns].count("class") == 1 and columns[-1][1] == "class", name
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.csv"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(",".join(f"{n}:{t}" for n, t in columns) + "\n")
        for r in rows:
            assert len(r) == len(columns), (name, r)
            f.write(",".join(r) + "\n")
    classes = Counter(r[-1] for r in rows)
    n_num = sum(t == "num" for _, t in columns)
    n_cat = sum(t == "cat" for _, t in columns)
    missing = sum(v == "?" for r in rows for v in r)
    majority = max(classes.values()) / max(1, len(rows))
    print(f"  {path.relative_to(ROOT)}: {len(rows)} records, {n_num} numeric + {n_cat} categorical attributes, "
          f"{len(classes)} classes (majority {100 * majority:.1f}%), {missing} missing values")


def categorical(name: str, text: str, attrs: list[str], class_col: str, numeric: set[str] = frozenset()) -> None:
    """All-categorical table (optionally with some numeric columns); the class column is moved last."""
    ci = attrs.index(class_col)
    order = [i for i in range(len(attrs)) if i != ci] + [ci]
    columns = [(attrs[i], "class" if i == ci else "num" if attrs[i] in numeric else "cat") for i in order]
    rows = []
    for line in text.splitlines():
        vals = [clean(v) for v in line.strip().split(",")]
        if len(vals) != len(attrs):
            continue
        rows.append([vals[i] for i in order])
    write(name, columns, rows)


def mushroom() -> None:
    z = download(f"{BASE}/73/mushroom.zip", "mushroom.zip")
    categorical("mushroom", zip_member(z, "agaricus-lepiota.data").decode(), MUSHROOM_ATTRS, "class")


def chess() -> None:
    z = download(f"{BASE}/22/chess+king+rook+vs+king+pawn.zip", "chess.zip")
    categorical("chess", zip_member(z, "kr-vs-kp.data").decode(), CHESS_ATTRS, "class")


def connect4() -> None:
    z = download(f"{BASE}/26/connect+4.zip", "connect4.zip")
    categorical("connect4", unlzw(zip_member(z, "connect-4.data.Z")).decode(), CONNECT4_ATTRS, "class")


def adult() -> None:
    """Train + test. fnlwgt (a census sampling weight, not a property of the person) is dropped."""
    z = download(f"{BASE}/2/adult.zip", "adult.zip")
    keep = [i for i, c in enumerate(ADULT_COLS) if c != "fnlwgt"]
    columns = [(ADULT_COLS[i], "class" if ADULT_COLS[i] == "income" else
                "num" if ADULT_COLS[i] in ADULT_NUMERIC else "cat") for i in keep]
    rows = []
    for member in ("adult.data", "adult.test"):
        for line in zip_member(z, member).decode().splitlines():
            vals = [clean(v) for v in line.split(",")]
            if len(vals) != len(ADULT_COLS):
                continue
            vals[-1] = vals[-1].rstrip(".")     # the test file writes "<=50K."
            rows.append([vals[i] for i in keep])
    write("adult", columns, rows)


def tictactoe() -> None:
    z = download(f"{BASE}/101/tic+tac+toe+endgame.zip", "tictactoe.zip")
    categorical("tictactoe", zip_member(z, "tic-tac-toe.data").decode(), TICTACTOE_ATTRS, "class")


def votes() -> None:
    z = download(f"{BASE}/105/congressional+voting+records.zip", "votes.zip")
    categorical("votes", zip_member(z, "house-votes-84.data").decode(), VOTES_ATTRS, "party")


def splice() -> None:
    z = download(f"{BASE}/69/molecular+biology+splice+junction+gene+sequences.zip", "splice.zip")
    columns = [(f"p{i + 1:02d}", "cat") for i in range(60)] + [("class", "class")]
    rows = []
    for line in zip_member(z, "splice.data").decode().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3 or len(parts[2]) != 60:
            continue
        rows.append(list(parts[2]) + [parts[0]])
    write("splice", columns, rows)


def nursery() -> None:
    z = download(f"{BASE}/76/nursery.zip", "nursery.zip")
    categorical("nursery", zip_member(z, "nursery.data").decode(), NURSERY_ATTRS, "class")


def letter() -> None:
    z = download(f"{BASE}/59/letter+recognition.zip", "letter.zip")
    categorical("letter", zip_member(z, "letter-recognition.data").decode(), LETTER_ATTRS, "letter",
                numeric=set(LETTER_ATTRS[1:]))


def krkopt() -> None:
    """Board files (a-h) stay categorical, ranks (1-8) are numeric."""
    z = download(f"{BASE}/23/chess+king+rook+vs+king.zip", "krkopt.zip")
    categorical("krkopt", zip_member(z, "krkopt.data").decode(), KRKOPT_ATTRS, "result",
                numeric={"wk-rank", "wr-rank", "bk-rank"})


def poker() -> None:
    """Train + test. Each card keeps its two columns: suit (categorical, 1-4) and rank (numeric, 1-13)."""
    z = download(f"{BASE}/158/poker+hand.zip", "poker.zip")
    columns = [c for k in range(1, 6) for c in ((f"suit{k}", "cat"), (f"rank{k}", "num"))] + [("hand", "class")]
    rows = []
    for member in ("poker-hand-training-true.data", "poker-hand-testing.data"):
        for line in zip_member(z, member).decode().splitlines():
            v = line.strip().split(",")
            if len(v) == 11:
                rows.append(v)
    write("poker", columns, rows)


def covertype() -> None:
    """The 4 wilderness-area and 40 soil-type indicator columns are folded back into two categorical attributes."""
    z = download(f"{BASE}/31/covertype.zip", "covertype.zip")
    columns = [(a, "num") for a in COVTYPE_NUMERIC] + [("wilderness", "cat"), ("soil", "cat"), ("cover", "class")]
    rows = []
    for line in gzip.decompress(zip_member(z, "covtype.data.gz")).decode().splitlines():
        r = line.split(",")
        if len(r) != 55:
            continue
        wild = [k + 1 for k in range(4) if r[10 + k] == "1"]
        soil = [k + 1 for k in range(40) if r[14 + k] == "1"]
        assert len(wild) == 1 and len(soil) == 1, line
        rows.append(r[:10] + [str(wild[0]), str(soil[0]), r[54]])
    write("covertype", columns, rows)


def census() -> None:
    """Train + test. The instance weight is dropped; coded columns (industry, occupation, year, ...) are categorical."""
    z = download(f"{BASE}/117/census+income+kdd.zip", "census_kdd.zip")
    tar = tarfile.open(fileobj=io.BytesIO(zip_member(z, "census.tar.gz")))
    keep = [i for i, c in enumerate(CENSUS_COLS) if c != "instance-weight"]
    columns = [(CENSUS_COLS[i], "class" if CENSUS_COLS[i] == "income" else
                "num" if CENSUS_COLS[i] in CENSUS_NUMERIC else "cat") for i in keep]
    rows = []
    for member in ("census-income.data", "census-income.test"):
        for line in tar.extractfile(member).read().decode().splitlines():
            vals = [clean(v) for v in line.split(",")]
            if len(vals) != len(CENSUS_COLS):
                continue
            vals[-1] = vals[-1].rstrip(".").replace("_", "")     # "- 50000." / "50000+."
            rows.append([vals[i] for i in keep])
    write("census", columns, rows)


DATASETS = {
    "mushroom": mushroom, "chess": chess, "connect4": connect4, "adult": adult, "tictactoe": tictactoe,
    "votes": votes, "splice": splice, "nursery": nursery, "letter": letter, "krkopt": krkopt, "poker": poker,
    "covertype": covertype, "census": census,
}


def main() -> None:
    names = sys.argv[1:] or list(DATASETS)
    for n in names:
        if n not in DATASETS:
            sys.exit(f"unknown dataset {n}; choose from {', '.join(DATASETS)}")
        print(n)
        DATASETS[n]()


if __name__ == "__main__":
    main()
