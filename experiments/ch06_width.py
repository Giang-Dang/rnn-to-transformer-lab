"""Chapter 6: the same widths as chapter 5, with and without attention.

Chapter 5 shrank the context vector and watched the model fail, long sentences
first. This script runs that same sweep twice: once with chapter 5's model,
where the decoder sees 2 * n_hidden numbers however long the sentence was, and
once with the model of Bahdanau et al., where it sees one annotation per source
position and chooses among them.

Everything else is held identical on purpose - same corpus, same split, same
seed, same batch size, same epoch count, same learning rate, same reversed
source, same greedy decoding, same exact-match scoring. The fixed-vector rows
reproduce chapter 5's table figure for figure, which is the check that the two
chapters' numbers may be set beside each other at all.

Read the `params` column with the accuracy column. At equal n_hidden the
attention model is the larger of the two, so an equal-n_hidden row is not a
fair fight on its own. The comparison that survives that objection is the one
between rows: attention at n_hidden 32 against the fixed vector at 128.

Run: python experiments/ch06_width.py
"""

from __future__ import annotations

import time

from rnn_to_transformer_lab import attention, seq2seq
from rnn_to_transformer_lab.determinism import describe_environment, utf8_stdout
from rnn_to_transformer_lab.seq2seq import EPOCHS, N_TEST, N_TRAIN
from rnn_to_transformer_lab.toy_corpus import disjoint_splits, vocabularies

WIDTHS = (16, 32, 64)
SEED = 0
#: Chapter 5's threshold, repeated rather than re-derived: the grammar makes one
#: clause 5 to 11 source tokens and two clauses 11 to 23, so this is the clause
#: boundary and it was fixed before any result was seen.
SHORT = 11


def measure(module, model, test_pairs, source_vocab, target_vocab):
    accuracy, hits = module.greedy_accuracy(
        model, test_pairs, source_vocab, target_vocab, reverse_source=True
    )
    short = [hit for length, hit in hits if length <= SHORT]
    long = [hit for length, hit in hits if length > SHORT]
    return accuracy, sum(short) / len(short), sum(long) / len(long)


def main() -> None:
    utf8_stdout()
    started = time.perf_counter()
    print(describe_environment())

    source_vocab, target_vocab = vocabularies()
    train_pairs, test_pairs = disjoint_splits(N_TRAIN, N_TEST, seed=5)
    print(f"train {len(train_pairs)} test {len(test_pairs)} epochs {EPOCHS} "
          f"source reversed, greedy decoding")
    print()

    print("d_hidden  model      params   loss     exact    short    long")
    for width in WIDTHS:
        for name, module, builder in (
            ("fixed", seq2seq, seq2seq.train_one),
            ("attention", attention, attention.train_one),
        ):
            model, losses = builder(
                SEED, reverse_source=True, n_hidden=width,
                train_pairs=train_pairs,
            )
            exact, short, long = measure(
                module, model, test_pairs, source_vocab, target_vocab
            )
            n_params = sum(p.numel() for p in model.parameters())
            print(
                f"{width:<9} {name:<10} {n_params:<8} "
                f"{sum(losses[-20:]) / 20:<8.4f} {exact:<8.4f} "
                f"{short:<8.4f} {long:.4f}"
            )

    print()
    n_short = sum(1 for p in test_pairs if len(p.source) <= SHORT)
    print(f"short = source of at most {SHORT} tokens ({n_short} of "
          f"{len(test_pairs)}), long = the rest")
    print(f"elapsed {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
