"""Chapter 5: reversing the source sentence, the paper's one data-side trick.

Section 3.3 of Sutskever et al. reports that reversing the words of the source
sentence, and only the source, dropped test perplexity from 5.8 to 4.7 and
raised BLEU from 25.9 to 30.6. Their explanation is about distance: reversing
leaves the average distance between corresponding words unchanged, but it puts
the first few source words next to the first few target words, so the problem's
minimal time lag is greatly reduced.

That argument predicts something sharper than "it helps", and this script tests
the sharper version: the gain should land on long sentences, because on a short
one every word is already near enough.

Three seeds per condition, and the per-seed rows are printed rather than only
the mean. A single pair of runs is not a measurement here: the spread between
seeds at this scale is wider than the effect being measured, which the ensemble
script demonstrates from the other direction.

Run: python experiments/ch05_reverse.py
"""

from __future__ import annotations

import time

from rnn_to_transformer_lab.determinism import describe_environment, utf8_stdout
from rnn_to_transformer_lab.seq2seq import (
    EPOCHS,
    N_TEST,
    N_TRAIN,
    greedy_accuracy,
    train_one,
)
from rnn_to_transformer_lab.toy_corpus import disjoint_splits, vocabularies

SEEDS = (0, 1, 2)
SHORT = 11


def main() -> None:
    utf8_stdout()
    started = time.perf_counter()
    print(describe_environment())

    source_vocab, target_vocab = vocabularies()
    train_pairs, test_pairs = disjoint_splits(N_TRAIN, N_TEST, seed=5)
    print(f"train {len(train_pairs)} test {len(test_pairs)} epochs {EPOCHS} "
          f"d_hidden 128, greedy decoding")
    print()

    summary: dict[bool, list[tuple[float, float, float]]] = {False: [], True: []}
    print("source     seed  loss     exact    short    long")
    for reverse in (False, True):
        for seed in SEEDS:
            model, losses = train_one(
                seed, reverse_source=reverse, train_pairs=train_pairs
            )
            accuracy, hits = greedy_accuracy(
                model, test_pairs, source_vocab, target_vocab, reverse_source=reverse
            )
            short = [hit for length, hit in hits if length <= SHORT]
            long = [hit for length, hit in hits if length > SHORT]
            row = (accuracy, sum(short) / len(short), sum(long) / len(long))
            summary[reverse].append(row)
            print(
                f"{'reversed' if reverse else 'raw':<10} {seed:<5} "
                f"{sum(losses[-20:]) / 20:<8.4f} "
                f"{row[0]:<8.4f} {row[1]:<8.4f} {row[2]:.4f}"
            )

    print()
    print("mean over seeds:")
    print("source     exact    short    long")
    means = {}
    for reverse in (False, True):
        rows = summary[reverse]
        means[reverse] = tuple(sum(r[i] for r in rows) / len(rows) for i in range(3))
        print(
            f"{'reversed' if reverse else 'raw':<10} "
            f"{means[reverse][0]:<8.4f} {means[reverse][1]:<8.4f} {means[reverse][2]:.4f}"
        )

    print()
    print("reversed minus raw:")
    print(f"  exact  {means[True][0] - means[False][0]:+.4f}")
    print(f"  short  {means[True][1] - means[False][1]:+.4f}")
    print(f"  long   {means[True][2] - means[False][2]:+.4f}")

    print()
    print(f"elapsed {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
