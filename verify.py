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
#:
#: Chapters 1 to 4 hold to 60 seconds per experiment, because each of their
#: scripts probes a computation that is fixed before the script starts. From
#: chapter 5 a script trains several models in order to compare them, and 60
#: seconds stops being a property of the script and becomes a property of how
#: many models it needs: the reversal table is six trainings, and no amount of
#: care makes six trainings fit one training's budget. So the rule from here is
#: 60 seconds per model trained, with a floor of 30 for a script that trains
#: none. What protects the reader is BUDGET_TOTAL below, and chapter 6 is where
#: that number finally had to move; the note on it says why and by how much.
ITEMS: tuple[tuple[str, str, list[str], float], ...] = (
    ("chapter 1 verify", "ch01", ["-c", "from rnn_to_transformer_lab import verify; verify()"], 30.0),
    ("chapter 2 verify", "ch02", ["-c", "from rnn_to_transformer_lab.ch02_symptoms import verify; verify()"], 300.0),
    ("chapter 3 tests", "ch03", ["-m", "pytest", "-q", "tests/test_ch03.py"], 120.0),
    ("experiments/ch03_decay.py", "ch03", ["experiments/ch03_decay.py"], 60.0),
    ("experiments/ch03_nonnormal.py", "ch03", ["experiments/ch03_nonnormal.py"], 60.0),
    ("experiments/ch03_surface.py", "ch03", ["experiments/ch03_surface.py"], 60.0),
    ("experiments/ch03_clipping.py", "ch03", ["experiments/ch03_clipping.py"], 60.0),
    ("experiments/ch03_regularizer.py", "ch03", ["experiments/ch03_regularizer.py"], 60.0),
    ("chapter 4 tests", "ch04", ["-m", "pytest", "-q", "tests/test_ch04.py"], 120.0),
    ("experiments/ch04_cec.py", "ch04", ["experiments/ch04_cec.py"], 60.0),
    ("experiments/ch04_conflict.py", "ch04", ["experiments/ch04_conflict.py"], 60.0),
    ("experiments/ch04_flow.py", "ch04", ["experiments/ch04_flow.py"], 60.0),
    ("experiments/ch04_truncation.py", "ch04", ["experiments/ch04_truncation.py"], 60.0),
    ("experiments/ch04_params.py", "ch04", ["experiments/ch04_params.py"], 60.0),
    ("experiments/ch04_adding.py", "ch04", ["experiments/ch04_adding.py"], 60.0),
    ("chapter 5 tests", "ch05", ["-m", "pytest", "-q", "tests/test_ch05.py"], 120.0),
    ("experiments/ch05_corpus.py", "ch05", ["experiments/ch05_corpus.py"], 30.0),
    ("experiments/ch05_bottleneck.py", "ch05", ["experiments/ch05_bottleneck.py"], 360.0),
    ("experiments/ch05_reverse.py", "ch05", ["experiments/ch05_reverse.py"], 360.0),
    ("experiments/ch05_search.py", "ch05", ["experiments/ch05_search.py"], 180.0),
    ("chapter 6 tests", "ch06", ["-m", "pytest", "-q", "tests/test_ch06.py"], 120.0),
    ("experiments/ch06_gradient.py", "ch06", ["experiments/ch06_gradient.py"], 30.0),
    ("experiments/ch06_alignment.py", "ch06", ["experiments/ch06_alignment.py"], 60.0),
    ("experiments/ch06_width.py", "ch06", ["experiments/ch06_width.py"], 360.0),
    ("experiments/ch06_encoder.py", "ch06", ["experiments/ch06_encoder.py"], 300.0),
)

#: Raised from 600 to 900 in the chapter 6 session, from the measurement rather
#: than because a run went over.
#:
#: 600 was set when this repo ended at chapter 3 and nothing in it trained a
#: translation model. Chapters 5 and 6 each add several trainings and the
#: number was never re-derived: it survived chapter 5 with about forty seconds
#: to spare on one machine, which is not headroom, it is luck. What makes that
#: visible is the spread between runs rather than any single total. Three whole
#: runs at tag ch05 came in at 557.52s, 495.51s and 487.27s - seventy seconds
#: of range on identical code, because these are wall-clock numbers from a
#: laptop that is also doing other things.
#:
#: Measured at tag ch06 against the 487.27s baseline: chapter 6 adds 5.92s of
#: tests, 0.34s for the gradient profile, 22.41s for the alignment run, 96.64s
#: for the encoder ablation and the width sweep on top, for a total near 715s.
#: 900 leaves roughly a fifth of the budget spare, which is the spread above
#: with room over.
#:
#: What is deliberately *not* changed is what the budget is for. Decision 13
#: promises the reader that every experiment finishes on a laptop with no
#: graphics card in minutes, and 900 seconds is still minutes. Before raising
#: it again, the question to ask is not whether the run fits but whether a
#: reader would still sit through one.
BUDGET_TOTAL = 900.0


def run(label: str, argv: list[str], budget: float) -> tuple[bool, bool, float]:
    """Run one item. Returns (passed, inside budget, elapsed).

    `encoding` is not optional and not cosmetic. From chapter 5 the experiments
    print Vietnamese, and `text=True` alone decodes the child's stdout with the
    locale encoding, which is cp1252 on Windows. The child having already been
    told to *write* UTF-8 does not help: this is the reading end, and it fails
    with a UnicodeDecodeError raised on a reader thread, so the traceback names
    `threading` and `subprocess` and nothing in this repo.
    """
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, *argv], cwd=ROOT, capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=False,
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
