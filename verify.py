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
    ("chapter 7 tests", "ch07", ["-m", "pytest", "-q", "tests/test_ch07.py"], 120.0),
    ("experiments/ch07_scaling.py", "ch07", ["experiments/ch07_scaling.py"], 30.0),
    ("experiments/ch07_position.py", "ch07", ["experiments/ch07_position.py"], 30.0),
    ("experiments/ch07_mask.py", "ch07", ["experiments/ch07_mask.py"], 120.0),
    # Two models, but the second is trained for 56 epochs against the shared
    # recipe's 14, which is four trainings' worth of work in one row of the
    # table. Decision 37's "60 seconds per model" is a rule about models at the
    # shared epoch count; a script that deliberately trains past it scales with
    # the epochs, and 180 is roughly three times the 65.18s measured at tag
    # ch07. The chapter's whole argument is that the extra epochs are the
    # finding, so this is not a budget that can be bought back by training less.
    ("experiments/ch07_corpus.py", "ch07", ["experiments/ch07_corpus.py"], 180.0),
    ("chapter 8 tests", "ch08", ["-m", "pytest", "-q", "tests/test_ch08.py"], 120.0),
    ("experiments/ch08_flops.py", "ch08", ["experiments/ch08_flops.py"], 30.0),
    # A timing script, so its budget is the one number here that is not about
    # this machine: a laptop with fewer cores or a slower memory bus can be
    # several times slower on the same code without anything being wrong.
    ("experiments/ch08_clock.py", "ch08", ["experiments/ch08_clock.py"], 180.0),
    # Twelve trainings each: four configurations at three seeds. Decision 37's
    # "60 seconds per model trained" would allow 720s for either of these, which
    # is not a budget so much as an absence of one, so both are set at roughly
    # twice their measured time instead. Three seeds rather than one is not
    # optional here and cannot be traded away for budget: the whole finding of
    # the norm table is that the spread within one configuration is wider than
    # the gap between two, and one seed cannot say that.
    ("experiments/ch08_norm.py", "ch08", ["experiments/ch08_norm.py"], 360.0),
    ("experiments/ch08_recipe.py", "ch08", ["experiments/ch08_recipe.py"], 360.0),
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
#:
#: Chapter 7 did not need it moved, which is worth recording because the SPEC
#: warned that it might. Its five items add about 95s: 2.4s of tests, 1.5s for
#: the scaling probe, 0.1s for the positional one, 26.1s for the mask, and
#: 65.2s for the corpus table. That is well inside the fifth of the budget left
#: spare at tag ch06. Three of the five train nothing at all - the tests, the
#: scaling probe and the positional one - and the two that do train are the
#: whole of the 91s. A chapter whose claims are mostly about a computation
#: rather than about a trained model is cheap to verify, and chapter 7's are.
#:
#: Raised from 900 to 1300 in the chapter 8 session, from the measurement, and
#: this is the second raise so the reasoning has to be better than "it did not
#: fit". Measured at tag ch08: the whole run is 1132.13s, of which chapter 8 is
#: 365.46s - 2.65s of tests, 2.03s for the FLOP tables, 20.87s for the clock,
#: 168.37s for the layer-norm study and 171.54s for the regularizers. Every
#: individual item passed its own budget; only the total was over.
#:
#: Two of chapter 8's items are 93% of its cost and both are twelve trainings,
#: four configurations at three seeds. That third seed is not padding and
#: cannot be traded back for budget: the finding of the layer-norm table is
#: that the spread *within* one configuration (0.3800 to 0.6500 exact match) is
#: wider than the gap *between* configurations, and a one-seed table asserts an
#: effect that three seeds show is noise. Decision 44 is the same lesson one
#: level down. Buying the budget back there would buy back a wrong chapter.
#:
#: On the size of the margin. Chapters 1 to 7 came in at 766.67s in this run
#: against the 663.51s recorded at tag ch07 on identical code - 103 seconds of
#: spread, wider than the 70s the ch06 note measured, because this run shared
#: the machine with other work. A budget whose headroom is narrower than its
#: own noise is not a budget, which is the argument the ch06 note made for 900
#: and it has not changed. 1300 leaves about 170s over the contended
#: measurement and around a fifth over an uncontended one.
#:
#: What has NOT been re-decided is what the budget is for, and this is now the
#: question rather than a formality. Decision 13 promises a reader with no
#: graphics card that every experiment finishes in minutes. 900s was fifteen
#: minutes; 1300s is nearly twenty-two, and the honest thing to say is that
#: this is approaching the edge of what "minutes" can be stretched to cover.
#: The next chapter that needs it moved should not move it. It should either
#: make an experiment cheaper - the open item on `torch.nn.LSTM` in the book's
#: SPEC is the largest single lever nobody has pulled, and chapter 8's own
#: clock table now measures that lever at 50 to 95 times on this hardware - or
#: split `verify.py` so a reader can check one chapter without running all of
#: them, which `--only` already allows and the README does not yet teach.
BUDGET_TOTAL = 1300.0


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
