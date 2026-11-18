"""Runs every correctness suite:

  check_data.py       table loader, stratified splits, metrics
  test_dtree.py       decision tree (textbook examples, Python reference, scikit-learn)
  test_baselines.py   naive Bayes, k-nearest neighbors, logistic regression

usage: python scripts/run_tests.py [--bin-dir bin] [--quick]
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    args = sys.argv[1:]
    data_args = [a for a in args if a != "--quick"]      # check_data.py has no quick mode
    failed = []
    for script, extra in (("check_data.py", data_args), ("test_dtree.py", args), ("test_baselines.py", args)):
        print(f"\n===== {script}", flush=True)
        if subprocess.run([sys.executable, str(HERE / script)] + extra).returncode != 0:
            failed.append(script)
    print("\n===== " + ("all suites passed" if not failed else "FAILED: " + ", ".join(failed)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
