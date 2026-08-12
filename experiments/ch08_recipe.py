"""Chapter 8: dropout and label smoothing, on a corpus that cannot use them.

Section 5.4 applies both to every model in the paper, and table 3 row (D)
measures what each is worth on WMT'14 English-German:

    P_drop 0.0  ->  PPL 5.77, BLEU 24.6      (base: PPL 4.92, BLEU 25.8)
    P_drop 0.2  ->  PPL 4.95, BLEU 25.5
    eps_ls 0.0  ->  PPL 4.67, BLEU 25.3
    eps_ls 0.2  ->  PPL 5.47, BLEU 25.7

Both are load-bearing there, and the label-smoothing row is the interesting one
because its two columns move in opposite directions: turning it off *improves*
perplexity and costs half a point of BLEU, which is precisely what section 5.4
says it does.

This script asks the same question of this book's corpus, and the chapter
predicts the answer before running it. Dropout is a defence against
overfitting; chapter 7 measured that this model does not overfit here, scoring
no better on data it trained on than on data it did not. A defence against a
thing that is not happening can only cost. So the prediction is that both
regularizers hurt, and that this is a fact about a finite grammar of 6000
sentences rather than a disagreement with the paper.

A 2x2 rather than a one-at-a-time sweep, so the interaction is visible: the
paper never ran either one alone.

**The loss column is not comparable across the label-smoothing rows** and is
printed anyway, with this warning attached. A smoothed target is unreachable by
a confident model, so a perfectly trained smoothed model still carries loss;
comparing 0.05 under smoothing with 0.05 without it compares two different
quantities. Exact match is the column to read across rows.

Run: python experiments/ch08_recipe.py
"""

from __future__ import annotations

import time

from rnn_to_transformer_lab.determinism import (
    describe_environment,
    seed_everything,
    utf8_stdout,
)
from rnn_to_transformer_lab.seq2seq import N_TEST, N_TRAIN
from rnn_to_transformer_lab.toy_corpus import disjoint_splits, vocabularies
from rnn_to_transformer_lab.transformer import greedy_accuracy, train_one

D_MODEL = 24
N_HEADS = 4
N_LAYERS = 2
SEEDS = (0, 1, 2)
#: (dropout, label smoothing). The first is this book's recipe from chapter 5
#: on; the last is the paper's base model, section 5.4.
GRID = ((0.0, 0.0), (0.1, 0.0), (0.0, 0.1), (0.1, 0.1))


def smoothing_floor(n_classes: int, epsilon: float) -> float:
    """The loss a *perfect* model still pays under label smoothing.

    Cross-entropy against the smoothed target q is minimised when the model
    predicts q exactly, and its value there is the entropy of q rather than
    zero. With mass (1 - eps + eps/K) on the true token and eps/K on each of
    the other K - 1,

        H(q) = -(1 - eps + eps/K) log(1 - eps + eps/K) - (K-1)(eps/K) log(eps/K)

    Subtracting this is what makes a smoothed loss and an unsmoothed one the
    same quantity again. Without it the two columns are not on one scale and
    the whole table says only that smoothing raises a number, which it does by
    construction.
    """
    if epsilon == 0.0:
        return 0.0
    from math import log

    true_mass = 1.0 - epsilon + epsilon / n_classes
    other = epsilon / n_classes
    return -(true_mass * log(true_mass) + (n_classes - 1) * other * log(other))


def main() -> None:
    utf8_stdout()
    started = time.perf_counter()
    print(describe_environment())
    seed_everything(0)

    source_vocab, target_vocab = vocabularies()
    _, test_pairs = disjoint_splits(N_TRAIN, N_TEST, seed=5)

    print()
    print("section 5.4's two regularizers, 2x2, on this book's corpus")
    print(f"d_model = {D_MODEL}, {N_HEADS} heads, {N_LAYERS} layers, 14 epochs")
    print(f"{len(SEEDS)} seeds per row")
    print(f"target vocabulary is {len(target_vocab)} classes, which sets the floor")
    print()
    print("P_drop  eps_ls  loss      floor     excess    exact     min      max      long")
    for dropout, smoothing in GRID:
        scores, shorts, longs, final_losses = [], [], [], []
        for seed in SEEDS:
            model, losses = train_one(
                seed, reverse_source=True, d_model=D_MODEL, n_heads=N_HEADS,
                n_layers=N_LAYERS, dropout=dropout, label_smoothing=smoothing,
            )
            model.eval()
            exact, hits = greedy_accuracy(
                model, test_pairs, source_vocab, target_vocab, reverse_source=True
            )
            short = [hit for length, hit in hits if length <= 11]
            long = [hit for length, hit in hits if length > 11]
            scores.append(exact)
            shorts.append(sum(short) / len(short))
            longs.append(sum(long) / len(long))
            final_losses.append(sum(losses[-20:]) / 20)
        mean = lambda xs: sum(xs) / len(xs)
        floor = smoothing_floor(len(target_vocab), smoothing)
        print(
            f"{dropout:<7.1f} {smoothing:<7.1f} {mean(final_losses):<9.4f} "
            f"{floor:<9.4f} {mean(final_losses) - floor:<9.4f} "
            f"{mean(scores):<9.4f} {min(scores):<8.4f} {max(scores):<8.4f} "
            f"{mean(longs):.4f}"
        )

    print()
    print("model.eval() is called before scoring, which matters only for the")
    print("dropout rows: leaving it in training mode would drop units at decode")
    print("time too, and the row would measure a bug rather than a regularizer")

    print()
    print(f"elapsed {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
