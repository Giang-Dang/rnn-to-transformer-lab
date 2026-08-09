"""The one command that says whether this repo is good.

    conda activate rnn-to-transformer-lab
    python verify.py

It runs each chapter's verification in order, then the chapter 3 test suite,
then every chapter 3 experiment script, and checks each one finished inside its
time budget. A non-zero exit means the repo is not in a state any chapter may
be tagged against.

    python verify.py --only ch03      # just the chapter 3 items
    python verify.py --list           # names, budgets, nothing run

The time budget is part of the gate rather than a note in the README. The book
tells a reader with no graphics card that these finish in minutes, and a
promise nothing measures is a promise that quietly stops being true.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

#: (name, chapter, argv, budget in seconds).
#:
#: The chapter 2 budget is the outlier. It trains ten plain RNNs from scratch
#: in numpy, the largest for 20000 samples of 100 steps, and it is by far the
#: slowest thing in this repo: 91s measured against 15s for everything else put
#: together. The budget is set at roughly three times the measured time so a
#: slower machine still passes. See the book's SPEC open items; if this grows
#: further the fix belongs to chapter 2, not to chapter 3.
ITEMS: tuple[tuple[str, str, list[str], float], ...] = (
    ("chapter 1 verify", "ch01", ["-c", "from rnn_to_transformer_lab import verify; verify()"], 30.0),
    ("chapter 2 verify", "ch02", ["-c", "from rnn_to_transformer_lab.ch02_symptoms import verify; verify()"], 300.0),
    ("chapter 3 tests", "ch03", ["-m", "pytest", "-q"], 120.0),
    ("experiments/ch03_decay.py", "ch03", ["experiments/ch03_decay.py"], 60.0),
    ("experiments/ch03_nonnormal.py", "ch03", ["experiments/ch03_nonnormal.py"], 60.0),
    ("experiments/ch03_surface.py", "ch03", ["experiments/ch03_surface.py"], 60.0),
    ("experiments/ch03_clipping.py", "ch03", ["experiments/ch03_clipping.py"], 60.0),
    ("experiments/ch03_regularizer.py", "ch03", ["experiments/ch03_regularizer.py"], 60.0),
)

BUDGET_TOTAL = 600.0


def run(label: str, argv: list[str], budget: float) -> tuple[bool, bool, float]:
    """Run one item. Returns (passed, inside budget, elapsed)."""
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, *argv], cwd=ROOT, capture_output=True, text=True, check=False
    )
    elapsed = time.perf_counter() - started
    ok = completed.returncode == 0
    in_budget = elapsed <= budget
    flag = "ok  " if ok else "FAIL"
    mark = "" if in_budget else f"  OVER BUDGET ({budget:.0f}s)"
    print(f"[{flag}] {label:<34} {elapsed:8.2f}s{mark}")
    if not ok:
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
    return ok, in_budget, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", metavar="CHAPTER", help="run only one chapter's items, e.g. ch03")
    parser.add_argument("--list", action="store_true", help="list the items and their budgets")
    args = parser.parse_args()

    items = [i for i in ITEMS if args.only is None or i[1] == args.only]
    if not items:
        print(f"no items for {args.only!r}; have {sorted({i[1] for i in ITEMS})}", file=sys.stderr)
        return 2

    if args.list:
        for label, chapter, _, budget in items:
            print(f"{chapter}  {label:<34} budget {budget:.0f}s")
        return 0

    print(f"repo: {ROOT}")
    print(f"python: {sys.executable}")
    print()

    failures: list[str] = []
    over_budget: list[str] = []
    total = 0.0

    for label, _, argv, budget in items:
        ok, in_budget, elapsed = run(label, argv, budget)
        total += elapsed
        if not ok:
            failures.append(label)
        if not in_budget:
            over_budget.append(f"{label} took {elapsed:.1f}s, budget {budget:.0f}s")

    print()
    print(f"total {total:.2f}s, budget {BUDGET_TOTAL:.0f}s")
    if total > BUDGET_TOTAL:
        over_budget.append(f"whole run took {total:.1f}s, budget {BUDGET_TOTAL:.0f}s")

    for line in over_budget:
        print(f"OVER BUDGET: {line}", file=sys.stderr)
    for line in failures:
        print(f"FAILED: {line}", file=sys.stderr)

    if failures or over_budget:
        return 1
    print("verify: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
