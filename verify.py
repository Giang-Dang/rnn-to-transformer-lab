"""The one command that says whether this repo is good.

    conda activate rnn-to-transformer-lab
    python verify.py

It runs every chapter's items in order - verifications, test suites and
experiment scripts, chapters 1 to 10 - and checks each one finished inside its
own time budget. A non-zero exit means the repo is not in a state any chapter
may be tagged against.

    python verify.py --only ch03      # just the chapter 3 items
    python verify.py --list           # names, budgets, nothing run

Time is part of the gate rather than a note in the README. The book tells a
reader with no graphics card that these finish in minutes, and a promise
nothing measures is a promise that quietly stops being true.

Two different failures, measured two different ways. **Per-item budgets are
hard**: one experiment taking several times what it should is a defect in that
experiment, and each budget carries 1.5x to 3x headroom so machine speed alone
cannot trip it. **The whole-run total is not a budget**, because wall clock
across a 20-minute run on a shared laptop is not a property of this repo; it is
reported against `TOTAL_TARGET` every run and only fails above `TOTAL_CEILING`.
The note on those two constants is where that reasoning lives.
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
#: none. What protects the reader is the whole-run total below, and chapter 6 is
#: where that number finally had to move; the note on it says why and by how
#: much. It was called BUDGET_TOTAL when this paragraph was written and is now
#: TOTAL_TARGET and TOTAL_CEILING.
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
    # Chapter 9 trains nothing at all. Its subject is models four orders of
    # magnitude past what this repo can run, so every claim it checks is a
    # count or a closed form, and the two experiment scripts do under a
    # hundredth of a second of real work each. The floor of decision 37 - 30
    # seconds for a script that trains no model - is what sets these budgets,
    # not any measurement of them.
    #
    # Worth stating rather than leaving as a coincidence: the whole-run note
    # below - BUDGET_TOTAL when this was written, TOTAL_TARGET and
    # TOTAL_CEILING now - said the next chapter needing it raised should not
    # raise it. Chapter 9 did not need it raised. It is the first chapter since
    # chapter 4 that adds no training time whatsoever.
    ("chapter 9 tests", "ch09", ["-m", "pytest", "-q", "tests/test_ch09.py"], 120.0),
    ("experiments/ch09_counts.py", "ch09", ["experiments/ch09_counts.py"], 30.0),
    ("experiments/ch09_laws.py", "ch09", ["experiments/ch09_laws.py"], 30.0),
    # Chapter 10 is the first chapter that needs data this repo cannot
    # generate, and the first whose experiments touch a real image dataset.
    # See decision 69 in the book's SPEC.
    #
    # The counting and equivariance scripts train nothing and take the floor of
    # decision 37. The CIFAR sweep trains 36 models - four training-set sizes,
    # three architectures, three seeds - and is set at roughly 1.5 times its
    # measured 292.54s rather than at decision 37's per-model rule, which would
    # allow 2160s here and is not a budget.
    #
    # That 292.54s is up from the 234.78s an earlier version of this sweep
    # measured, and the difference is not the restored 50,000-image row alone.
    # `standardize` now runs per training subset rather than once over the whole
    # split, because doing it once leaks statistics from images a small row
    # never sees; at 50,000 images that recomputation happens for every seed and
    # every architecture. The leak was worth more than the seconds.
    #
    # **This budget was 360s and did not have enough headroom.** 360 against
    # 292.54 is 1.23x, where every other item here carries 1.5x or better, so a
    # slower laptop would have failed an item that was not misbehaving. Raised
    # to 450, which is the same 1.5x the 360 was derived from.
    #
    # **This item needs the dataset on disk.** The first run downloads 163 MB,
    # which took 32 minutes on the machine this was written on and is nowhere
    # near any per-item budget. That is why the download is not part of the
    # timed work: fetch once with
    #   python -c "from rnn_to_transformer_lab.cifar import fetch; fetch()"
    # and every later run reads the cache. A reader who has not fetched gets a
    # failure here with that command in the message, which is the trade the
    # book's author accepted rather than have chapter 10 cite instead of
    # measure.
    ("chapter 10 tests", "ch10", ["-m", "pytest", "-q", "tests/test_ch10.py"], 120.0),
    ("experiments/ch10_counts.py", "ch10", ["experiments/ch10_counts.py"], 30.0),
    ("experiments/ch10_equivariance.py", "ch10", ["experiments/ch10_equivariance.py"], 30.0),
    ("experiments/ch10_cifar.py", "ch10", ["experiments/ch10_cifar.py"], 450.0),
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
#: **In the chapter 10 session `BUDGET_TOTAL` was retired.** It is replaced by
#: the two constants below, and everything above this line is the evidence for
#: how they are derived, so it stays.
#:
#: What broke. At tag ch10 the whole run was 1230.15s against 1300 on an idle
#: machine - 70s of headroom - while the chapters nobody had touched drifted
#: 94.33s between that run and the one recorded at tag ch09 (1001.60 to
#: 1095.93). The margin had become narrower than the noise, which is the ch06
#: note's own test for when a budget has stopped being one. Raising it a third
#: time was ruled out by the paragraph above. Leaving it was worse than it
#: looks: **a gate that fails on healthy code gets paid off rather than
#: obeyed**, and in one session chapter 10 paid twice - once by cutting epochs,
#: which starved its small-data rows and had to be reversed, and once by
#: dropping its 50,000-image row, which cost the chapter an actual measured
#: crossing point and left a bound in its place.
#:
#: So the number splits, because it was doing two jobs that need different
#: strictness. Reporting how long the run takes wants a number printed every
#: time and never fatal, since wall clock on a shared laptop is not a property
#: of this repo. Catching accretion - a chapter adding ten minutes that every
#: per-item budget waves through - wants a threshold noise cannot reach.
#:
#: The drift record, which is what the derivation rests on:
#:
#:     session  code compared                    totals            drift
#:     ch05     three runs at tag ch05           557.52/495.51/487.27   70.25s
#:     ch08     ch01-07, tag ch07 vs in session  663.51 -> 766.67      103.16s
#:     ch08     two whole runs, same session     ~1263 +/-             132.00s
#:     ch10     ch01-09, tag ch09 vs in session  1001.60 -> 1095.93     94.33s
#:     ch10     two whole runs, this shape       1305.35 / 1358.24      52.89s
#:
#: The last row is the pair H is taken from. One more, on a single item rather
#: than a whole run, because it shows the noise is not a property of the total:
#: ch10_cifar.py measured 292.54s run on its own and 251.10s inside a whole run,
#: 41.44s apart on byte-identical code, which is 14.2% and lands inside the
#: relative band below. That is also why every per-item budget here carries 1.5x
#: or more rather than something tight.
#:
#: Four points spanning 500-1300s cannot distinguish additive drift (70-132s
#: whatever the size) from proportional (9.4-15.5%), so the rule takes the worse
#: of both. That costs nothing and survives being wrong about which it is.
#:
#:     H       healthy total, idle machine, largest of >= 2 runs
#:     A       largest absolute drift on record            = 132.0 s
#:     R       largest relative drift on record            = 0.155
#:     D       = max(A, R * H)
#:     R_min   smallest regression worth catching          = 600 s
#:
#:     TOTAL_TARGET  = ceil_50( H + 1.5 * D )
#:     TOTAL_CEILING = ceil_50( (TOTAL_TARGET + H + R_min) / 2 )
#:
#: The 1.5 is because D is a sample maximum from four points and the true
#: maximum is above it. The ceiling is the midpoint between the worst a healthy
#: run reaches and the best a regressed one does, so it is equally far from
#: firing on noise and from letting a real regression through. That is the whole
#: content of the choice, and it is why the ceiling is not a round number picked
#: for looking like one.
#:
#: Substituted at tag ch10, with the 50,000-image row restored. Two whole runs
#: on an idle machine gave 1305.35s and 1358.24s, so H = 1358.24 - the rule says
#: the largest, because a single run underestimates and this pair is 52.89s
#: apart on identical code.
#:
#:     D       = max(132, 0.155 * 1358.24) = max(132, 210.53) = 210.53
#:     TARGET  = ceil_50(1358.24 + 315.79) = ceil_50(1674.03) = 1700
#:     CEILING = ceil_50((1700 + 1358.24 + 600) / 2) = ceil_50(1829.12) = 1850
#:
#: **Say the cost plainly rather than calling this a refactor.** Before, a run
#: over about 22 minutes failed; now one up to about 31 passes. The enforced
#: upper bound moved by nine minutes. What is bought with it is that the gate
#: stops being paid off in results, and decision 74 in the book's SPEC is the
#: receipt for what that was costing.
#:
#: **When this scheme stops working**, which is the part worth keeping. It needs
#: TARGET < CEILING < H + R_min, and that reduces to
#:
#:     H  <  R_min / (1.5 * R)  ~=  2580 s
#:
#: Past a healthy total of about 43 minutes no pair of thresholds can both avoid
#: firing on noise and catch a ten-minute experiment, because by then the noise
#: is larger than the regression. Today's H of 1358.24s uses 52.6% of it.
#: That is a measurement, not an allowance to spend: the answer at that point is
#: to make an experiment cheaper, not to move these numbers again. The book's
#: SPEC carries an open item on roughly 180-200s of duplicated training, which
#: is where to look first.
#:
#: **And a correction, because an earlier version of this note sent the next
#: session the wrong way.** It said chapter 8's clock table measures the fused
#: `torch.nn.LSTM` lever "at 50 to 95 times on this hardware". No table here
#: says that. `experiments/ch08_clock_canonical.txt` measures loop against
#: nn.LSTM at 2.4x to 8.1x across d = 128 and 512 at n = 32 to 128, and the
#: 25.2x the book quotes is one cell, forward-plus-backward at n = 512 - which
#: chapter 8's own prose is careful to scope as "measured, at one place". The
#: corpus chapters 5 to 7 train on is a few tokens per sentence, nowhere near
#: n = 512, so the lever is worth far less here than the retired sentence
#: implied. It is also not available: `LstmLayer` carries one bias vector of
#: 4*n_hidden where `nn.LSTM` carries two, so swapping changes printed
#: parameter counts in four chapters, and chapter 6's attention decoder cannot
#: be fused at all because its step input depends on the previous step's state.
TOTAL_TARGET = 1700.0
TOTAL_CEILING = 1850.0


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

    per_chapter: dict[str, float] = {}

    for label, chapter, argv, budget in items:
        ok, in_budget, elapsed = run(label, argv, budget)
        total += elapsed
        per_chapter[chapter] = per_chapter.get(chapter, 0.0) + elapsed
        if not ok:
            failures.append(label)
        if not in_budget:
            over_budget.append(f"{label} took {elapsed:.1f}s, budget {budget:.0f}s")

    print()
    over_ceiling: str | None = None
    if args.only is not None:
        # The whole-run thresholds describe the whole run. Printing them next to
        # one chapter's subtotal would be comparing a part against a limit set
        # for the sum, which is the kind of check that fires when nothing is
        # wrong - and this file's own argument for splitting the total is that
        # such a check trains its reader to ignore it.
        print(f"total {total:.2f}s for {args.only}")
    else:
        print(
            f"total {total:.2f}s, target {TOTAL_TARGET:.0f}s, "
            f"ceiling {TOTAL_CEILING:.0f}s"
        )
        if total > TOTAL_CEILING:
            over_ceiling = (
                f"whole run took {total:.1f}s, ceiling {TOTAL_CEILING:.0f}s"
            )
        if total > TOTAL_TARGET:
            # Only when something is over, because eleven extra lines on every
            # clean run is clutter, and "which chapter grew" is a question
            # nobody asks until one has.
            for name in sorted(per_chapter):
                print(f"  {name} {per_chapter[name]:8.2f}s")
        if total > TOTAL_TARGET and over_ceiling is None:
            print(
                f"note: {total - TOTAL_TARGET:.0f}s over the target and inside "
                "the ceiling, so this"
            )
            print(
                "      is not a failure. A busy machine does this and so does a"
            )
            print(
                "      chapter that has grown. Run again idle before calling it"
            )
            print("      a regression.")

    for line in over_budget:
        print(f"OVER BUDGET: {line}", file=sys.stderr)
    if over_ceiling is not None:
        # Deliberately not appended to over_budget: that loop prints
        # "OVER BUDGET", and the whole point of splitting the total is that it
        # is no longer a budget. Reusing the word would undo the change in the
        # one place a reader actually sees it.
        print(f"OVER CEILING: {over_ceiling}", file=sys.stderr)
        print(
            "  a healthy run measures about 1358s; no drift on record comes "
            "near this.",
            file=sys.stderr,
        )
        print(
            "  see the note on TOTAL_CEILING in verify.py before moving it.",
            file=sys.stderr,
        )
    for line in failures:
        print(f"FAILED: {line}", file=sys.stderr)

    if failures or over_budget or over_ceiling:
        return 1
    print("verify: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
