"""Chapter 5: what the toy English-Vietnamese corpus actually looks like.

No training here. This prints the grammar's output so that every later table in
the chapter is read against a corpus the reader has seen, and so that the two
claims the chapter makes about it are checked rather than asserted: that the
target sentence is usually longer than the source, and that the adjective
crosses the noun on the way from one language to the other.

Run: python experiments/ch05_corpus.py
"""

from __future__ import annotations

import time

from rnn_to_transformer_lab.determinism import describe_environment, utf8_stdout
from rnn_to_transformer_lab.seq2seq import N_TEST, N_TRAIN
from rnn_to_transformer_lab.toy_corpus import (
    disjoint_splits,
    statistics,
    vocabularies,
)

N_EXAMPLES = 3


def main() -> None:
    utf8_stdout()
    started = time.perf_counter()
    print(describe_environment())

    source_vocab, target_vocab = vocabularies()
    train_pairs, test_pairs = disjoint_splits(N_TRAIN, N_TEST, seed=5)

    print()
    print(f"source vocabulary {len(source_vocab)} types, "
          f"target {len(target_vocab)} types (both closed, both include "
          f"<pad> <sos> <eos>)")

    print()
    print("split  pairs  src min  src max  src mean  tgt min  tgt max  tgt mean")
    for label, pairs in (("train", train_pairs), ("test", test_pairs)):
        s = statistics(pairs)
        print(
            f"{label:<6} {s['pairs']:<6.0f} {s['src_min']:<8.0f} {s['src_max']:<8.0f} "
            f"{s['src_mean']:<9.3f} {s['tgt_min']:<8.0f} {s['tgt_max']:<8.0f} "
            f"{s['tgt_mean']:.3f}"
        )

    train_stats = statistics(train_pairs)
    print()
    print(f"target longer than source in {train_stats['longer_target']:.4f} of training pairs")

    overlap = {p.source for p in train_pairs} & {p.source for p in test_pairs}
    print(f"source sentences shared between train and test: {len(overlap)}")

    # Split by clause count rather than printing the first five of everything.
    # Two-clause sentences run past a hundred columns, which is wider than the
    # book's measure, so the chapter can only set the one-clause ones; taking
    # the first three of each keeps that selection stated rather than made by
    # eye afterwards.
    one_clause = [p for p in train_pairs if "and" not in p.source]
    two_clause = [p for p in train_pairs if "and" in p.source]

    print()
    print(f"first {N_EXAMPLES} one-clause sentences of the training split:")
    for pair in one_clause[:N_EXAMPLES]:
        print(f"  en  {' '.join(pair.source)}")
        print(f"  vi  {' '.join(pair.target)}")

    print()
    print(f"first {N_EXAMPLES} two-clause sentences of the training split:")
    for pair in two_clause[:N_EXAMPLES]:
        print(f"  en  {' '.join(pair.source)}")
        print(f"  vi  {' '.join(pair.target)}")

    print()
    print(f"elapsed {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
