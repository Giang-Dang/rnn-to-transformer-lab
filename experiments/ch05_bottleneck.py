"""Chapter 5: what the fixed-length context vector costs, measured by shrinking it.

Everything the decoder ever learns about the source sentence arrives as one
vector of 2 * n_hidden numbers, the encoder's final (c_T, h_T). This script
holds the corpus, the optimizer, the seed and the number of updates fixed and
moves only that width.

The split by source length is the part worth reading. A narrow context does not
fail uniformly: short sentences are already perfect at a width where long ones
are still at zero, which is what "the sentence has to fit in the vector" looks
like when you measure it instead of asserting it.

One seed per row. That is defensible here and not everywhere in this chapter:
the effect between the ends of this table is the whole range from 0 to most of
the way, while the reversal experiment next door measures a few points and
therefore repeats.

Run: python experiments/ch05_bottleneck.py
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

WIDTHS = (4, 8, 16, 32, 64, 128)
SEED = 0
#: Sentences of at most this many source tokens are counted "short". The
#: grammar makes one clause 5 to 11 tokens and two clauses 11 to 23, so this is
#: the clause boundary rather than a threshold chosen after seeing the results.
SHORT = 11


def main() -> None:
    utf8_stdout()
    started = time.perf_counter()
    print(describe_environment())

    source_vocab, target_vocab = vocabularies()
    train_pairs, test_pairs = disjoint_splits(N_TRAIN, N_TEST, seed=5)
    print(f"train {len(train_pairs)} test {len(test_pairs)} epochs {EPOCHS} "
          f"source reversed, greedy decoding")
    print()

    print("d_hidden  context  params   loss     exact    short    long")
    for width in WIDTHS:
        model, losses = train_one(
            SEED, reverse_source=True, n_hidden=width, train_pairs=train_pairs
        )
        accuracy, hits = greedy_accuracy(
            model, test_pairs, source_vocab, target_vocab, reverse_source=True
        )
        short = [hit for length, hit in hits if length <= SHORT]
        long = [hit for length, hit in hits if length > SHORT]
        n_params = sum(p.numel() for p in model.parameters())
        print(
            f"{width:<9} {2 * width:<8} {n_params:<8} "
            f"{sum(losses[-20:]) / 20:<8.4f} {accuracy:<8.4f} "
            f"{sum(short) / len(short):<8.4f} {sum(long) / len(long):.4f}"
        )

    print()
    print(f"short = source of at most {SHORT} tokens ({sum(1 for p in test_pairs if len(p.source) <= SHORT)} "
          f"of {len(test_pairs)}), long = the rest")
    print(f"elapsed {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
