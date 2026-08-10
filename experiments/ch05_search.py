"""Chapter 5: beam search and ensembling, the two things bolted on at decode time.

Both change what comes out of a trained model without changing the model, and
the paper reports them together in table 1, so they are measured together here
against the same three models rather than in two scripts that would each have
had to train their own.

What the paper found, in its own words: the system "performs well even with a
beam size of 1, and a beam of size 2 provides most of the benefits of beam
search", while an ensemble of five reversed LSTMs took the score from 30.59 to
34.81. So the expected shape is a small gain from widening the beam and a large
one from adding models, and that is a testable prediction rather than a summary.

The per-model rows come first on purpose. The spread between three models that
differ only in their seed is the context every other number in this chapter has
to be read against.

Run: python experiments/ch05_search.py
"""

from __future__ import annotations

import time

from rnn_to_transformer_lab.determinism import describe_environment, utf8_stdout
from rnn_to_transformer_lab.seq2seq import (
    EPOCHS,
    N_TEST,
    N_TRAIN,
    beam_decode,
    train_one,
)
from rnn_to_transformer_lab.toy_corpus import _pad, disjoint_splits, vocabularies

SEEDS = (0, 1, 2)
BEAMS = (1, 2, 5, 12)
SHORT = 11


def main() -> None:
    utf8_stdout()
    started = time.perf_counter()
    print(describe_environment())

    source_vocab, target_vocab = vocabularies()
    train_pairs, test_pairs = disjoint_splits(N_TRAIN, N_TEST, seed=5)
    print(f"train {len(train_pairs)} test {len(test_pairs)} epochs {EPOCHS} "
          f"d_hidden 128, source reversed")
    print()

    models = [
        train_one(seed, reverse_source=True, train_pairs=train_pairs)[0]
        for seed in SEEDS
    ]

    # Encode once. Every row below decodes the same 300 source sentences, and
    # re-encoding them per row is the kind of accident that makes two rows
    # incomparable.
    encoded = [
        (_pad([source_vocab.encode(list(pair.source)[::-1])]), target_vocab.encode(pair.target),
         len(pair.source))
        for pair in test_pairs
    ]

    def evaluate(subset, beam: int) -> tuple[float, float, float, float]:
        hits, short_hits, short_n, long_hits, long_n, lengths = 0, 0, 0, 0, 0, 0
        for source, reference, source_length in encoded:
            output = beam_decode(subset, source, beam=beam)
            ok = output == reference
            hits += ok
            lengths += len(output)
            if source_length <= SHORT:
                short_n += 1
                short_hits += ok
            else:
                long_n += 1
                long_hits += ok
        return (
            hits / len(encoded),
            short_hits / short_n,
            long_hits / long_n,
            lengths / len(encoded),
        )

    reference_length = sum(len(p.target) for p in test_pairs) / len(test_pairs)

    print("one model at a time, beam 2:")
    print("seed  exact    short    long")
    for seed, model in zip(SEEDS, models):
        exact, short, long, _ = evaluate([model], 2)
        print(f"{seed:<5} {exact:<8.4f} {short:<8.4f} {long:.4f}")

    print()
    print(f"beam size, model seed {SEEDS[0]} alone "
          f"(mean reference length {reference_length:.3f}):")
    print("beam  exact    short    long     mean output length")
    for beam in BEAMS:
        exact, short, long, mean_length = evaluate([models[0]], beam)
        print(f"{beam:<5} {exact:<8.4f} {short:<8.4f} {long:<8.4f} {mean_length:.3f}")

    print()
    print("ensemble size, beam 2:")
    print("models  exact    short    long")
    for k in (1, 2, 3):
        exact, short, long, _ = evaluate(models[:k], 2)
        print(f"{k:<7} {exact:<8.4f} {short:<8.4f} {long:.4f}")

    print()
    print(f"elapsed {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
