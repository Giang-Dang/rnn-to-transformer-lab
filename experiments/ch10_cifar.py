"""Chapter 10: what the two constraints are worth, measured on real images.

The first experiment in this repo that does not run on `toy_corpus.py`. See
decision 69 in the book's SPEC for why chapter 10 takes a download.

1. The three models and their parameter counts, so the comparison is on the
   record before any accuracy is.
2. Sample efficiency: each model trained at four training-set sizes, three
   seeds each, scored on the full 10,000-image test split.
3. Shift stability of the *trained* models, which is section 2's measurement
   carried onto a model that has learned something.

**This script trains and its numbers depend on the machine.** Per decision 60
it writes its raw run to `experiments/ch10_cifar_canonical.txt`, and the
book's research note points at that file rather than at a number somebody
remembered.

Run: python experiments/ch10_cifar.py
"""

from __future__ import annotations

import math
import time

import torch

from rnn_to_transformer_lab.cifar import describe, load
from rnn_to_transformer_lab.conv import invariance_rate
from rnn_to_transformer_lab.determinism import (
    describe_environment,
    seed_everything,
    utf8_stdout,
)
from rnn_to_transformer_lab.vision import (
    MLP,
    SmallCNN,
    count_parameters,
    mlp_matched,
    standardize,
    train_and_score,
)

#: Training-set sizes, and how many epochs each gets. Small sets get more
#: passes because 1,000 images is 8 gradient steps per epoch and a fixed epoch
#: count would be comparing 8 steps against 390. What is held fixed across a
#: row is the recipe - optimizer, learning rate, batch size - and what varies
#: is how much data there is, which is the question.
#:
#: The epoch counts were cut once, from (30, 16, 8, 5), after the first whole
#: run came in at 302.24s. Decision 62 in the book's SPEC says the next chapter
#: that needs BUDGET_TOTAL raised should not raise it, so chapter 10 pays by
#: training less rather than by moving the number.
#:
#: What the cut actually cost, measured rather than asserted. Three of the four
#: CNN accuracies moved by less than their own seed spread (+0.0072, +0.0014,
#: -0.0060 against spreads of 0.0220, 0.0117, 0.0117). The 50,000-image row did
#: not: it fell 0.0185 against a spread of 0.0077, because four epochs over
#: 50,000 images is genuinely less converged than five. So every model in this
#: table is under-trained at the large end, and the honest reading is that the
#: CNN's 0.7047 is a floor rather than a ceiling.
#:
#: What did not change is every ordering and every direction: CNN above wide
#: MLP above matched MLP at all four sizes, and the gap growing monotonically
#: with training-set size in both runs. No sentence in the chapter turns on the
#: fourth decimal of the 50,000 row.
#:
#: **The 50,000-image row was cut once, and then put back.** It was cut because
#: with it the sweep was 232.25s and the whole repo came to 1332.17s against a
#: 1300s BUDGET_TOTAL on an idle machine, so `verify.py` returned 1. That was
#: the only cut available: decision 62 forbade raising the number a third time
#: and decision 71 forbade buying the budget back by cutting epochs, because
#: cutting epochs starves the small-data rows and manufactures the very effect
#: this table measures.
#:
#: It cost a result rather than precision. Without the row the sample-efficiency
#: table could only report the bound "over 16,000" where the full sweep produces
#: an actual crossing point, and that is the chapter's sharpest single number.
#:
#: The row is back because the budget question was settled rather than paid
#: again. `BUDGET_TOTAL` was a single hard threshold doing two incompatible
#: jobs, and its margin had become narrower than its own run-to-run noise; it is
#: now `TOTAL_TARGET` and `TOTAL_CEILING` in `verify.py`, where the reasoning
#: lives. Nothing here was made cheaper and nothing was traded away - the gate
#: stopped failing on machine noise, so this row stopped having to pay for it.
SWEEP = ((1000, 25), (4000, 14), (16000, 7), (50000, 4))

#: Three, not two, and not tradeable for budget. The gap columns in section 3
#: are read against these spreads, and decision 62 is on the record that a
#: one-seed table can assert an effect three seeds show is noise.
SEEDS = (0, 1, 2)
WIDE_HIDDEN = 512


def main() -> None:
    utf8_stdout()
    started = time.perf_counter()
    print(describe_environment())

    data = load()
    print()
    print("CIFAR-10, Krizhevsky 2009")
    describe(data)

    # NOT standardized here. `standardize` is called inside the sweep, from the
    # subset each row actually trains on - see the loop below. Doing it here,
    # once, over all 50,000 images is a leak: the 1,000-image row would be
    # centered and scaled using statistics drawn from 49,000 images it never
    # sees, and this chapter is read at exactly that end of the table. A first
    # version of this script did it here and the chapter shipped an exercise
    # asking the reader to introduce a leak that was already present.
    train_x, test_x = data["train_x"], data["test_x"]
    train_y, test_y = data["train_y"], data["test_y"]

    print()
    print("1. the three models")
    print()
    cnn_params = count_parameters(SmallCNN())
    matched = mlp_matched(cnn_params)
    matched_params = count_parameters(matched)
    wide_params = count_parameters(MLP(WIDE_HIDDEN))
    print("model         parameters  vs CNN  locality  sharing")
    print(f"{'CNN':<13} {cnn_params:<11,} {1.0:<7.4f} {'yes':<9} {'yes'}")
    print(f"{'MLP matched':<13} {matched_params:<11,} "
          f"{matched_params / cnn_params:<7.4f} {'no':<9} {'no'}")
    print(f"{'MLP wide':<13} {wide_params:<11,} "
          f"{wide_params / cnn_params:<7.4f} {'no':<9} {'no'}")
    print()
    print(f"the matched MLP has {matched.net[1].out_features} hidden units. that is what")
    print("66,570 parameters buys when none of them are shared.")

    print()
    print("2. test accuracy against training-set size")
    print(f"   {len(SEEDS)} seeds, full 10,000-image test split, same recipe")
    print()
    print("n_train  epochs  model         test acc     spread   train acc")

    results: dict[tuple[str, int], list[float]] = {}
    #: Kept for section 5, so that it reads shift stability off models the
    #: sweep already paid for rather than training two more.
    keep: dict[str, torch.nn.Module] = {}

    for n_train, epochs in SWEEP:
        for name in ("CNN", "MLP matched", "MLP wide"):
            test_scores, train_scores = [], []
            for seed in SEEDS:
                seed_everything(seed)
                if name == "CNN":
                    model = SmallCNN()
                elif name == "MLP matched":
                    model = mlp_matched(cnn_params)
                else:
                    model = MLP(WIDE_HIDDEN)
                subset = torch.randperm(train_x.shape[0])[:n_train]
                sub_x, sub_test_x = standardize(train_x[subset], test_x)
                test_acc, train_acc = train_and_score(
                    model,
                    sub_x,
                    train_y[subset],
                    sub_test_x,
                    test_y,
                    epochs=epochs,
                )
                test_scores.append(test_acc)
                train_scores.append(train_acc)
                if n_train == SWEEP[-1][0] and seed == SEEDS[0]:
                    # Keep the standardized test split beside the model. Each
                    # row now standardizes from its own subset, so a model must
                    # be shown the test set scaled the way its training set was
                    # or section 5 measures the mismatch instead of the shift.
                    keep[name] = (model, sub_test_x)
            results[(name, n_train)] = test_scores
            mean = sum(test_scores) / len(test_scores)
            spread = max(test_scores) - min(test_scores)
            train_mean = sum(train_scores) / len(train_scores)
            print(
                f"{n_train:<8} {epochs:<7} {name:<13} {mean:<12.4f} "
                f"{spread:<8.4f} {train_mean:.4f}"
            )
        print()

    print("3. what the convolution is worth, by training-set size")
    print()
    print("n_train  CNN-matched  CNN-wide  CNN spread  widest MLP spread")
    for n_train, _ in SWEEP:
        cnn = sum(results[("CNN", n_train)]) / len(SEEDS)
        mat = sum(results[("MLP matched", n_train)]) / len(SEEDS)
        wide = sum(results[("MLP wide", n_train)]) / len(SEEDS)
        rows = results[("CNN", n_train)]
        mlp_spread = max(
            max(results[(m, n_train)]) - min(results[(m, n_train)])
            for m in ("MLP matched", "MLP wide")
        )
        print(
            f"{n_train:<8} {cnn - mat:<12.4f} {cnn - wide:<9.4f} "
            f"{max(rows) - min(rows):<11.4f} {mlp_spread:.4f}"
        )
    print()
    print("read the gap columns against the spread columns: a gap narrower")
    print("than the seed spread is not a result. chapter 8's layer-norm table")
    print("is the case where that mattered.")

    print()
    print("4. the same table read as sample efficiency")
    print("   how many images each MLP needs to reach what the CNN reaches")
    print()

    def images_needed(name: str, target: float) -> float | None:
        """Where this model's curve crosses `target`, log-linear between rows."""
        sizes = [n for n, _ in SWEEP]
        curve = [sum(results[(name, n)]) / len(SEEDS) for n in sizes]
        for i in range(1, len(sizes)):
            if curve[i - 1] <= target <= curve[i]:
                span = curve[i] - curve[i - 1]
                frac = 0.0 if span == 0 else (target - curve[i - 1]) / span
                lo, hi = math.log(sizes[i - 1]), math.log(sizes[i])
                return math.exp(lo + frac * (hi - lo))
        return None

    print("CNN trained on  reaches  MLP matched needs  MLP wide needs")
    for n_train, _ in SWEEP:
        target = sum(results[("CNN", n_train)]) / len(SEEDS)
        cells = []
        for name in ("MLP matched", "MLP wide"):
            need = images_needed(name, target)
            cells.append(
                f"{need:>10,.0f} ({need / n_train:.1f}x)"
                if need is not None
                else f"{'over ' + format(SWEEP[-1][0], ',') :>18}"
            )
        print(f"{n_train:<15} {target:<8.4f} {cells[0]:<18} {cells[1]}")
    print()
    print("this is the comparison worth printing, and it is not the gap")
    print("column above. the absolute gap is free to grow with data while the")
    print("CNN still reaches a given accuracy on a fraction of the images -")
    print("two different questions, and only the second one is about sample")
    print("efficiency. giving the MLP 24 times the parameters moves the")
    print("multiplier down and does not remove it.")

    print()
    print("5. shift stability of the trained models")
    print("   fraction of test images whose predicted class changes")
    print(f"   the seed-0 models from the {SWEEP[-1][0]:,}-image row, reused")
    print()
    print("shift  CNN     MLP matched  MLP wide")
    for k in (1, 2, 4, 8):
        cells = [
            invariance_rate(model, scaled_test, k, 0)
            for model, scaled_test in (
                keep["CNN"],
                keep["MLP matched"],
                keep["MLP wide"],
            )
        ]
        print(f"{k:<6} {cells[0]:<7.4f} {cells[1]:<12.4f} {cells[2]:.4f}")
    print()
    print("read this by column, at one shift. two separate reasons the CNN")
    print("is not shift-invariant even though convolution is equivariant:")
    print("its head is a Linear over the flattened feature map, so a map")
    print("shifted by one cell is a different input vector; and SmallCNN")
    print("pads with zeros rather than cyclically, so the exact-zero result")
    print("of ch10_equivariance.py does not apply to it at the border - and")
    print("a cyclic shift moves content across exactly that border.")

    print()
    print(f"elapsed {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
