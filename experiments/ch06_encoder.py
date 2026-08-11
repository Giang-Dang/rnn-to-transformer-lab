"""Chapter 6: the bidirectional encoder, and what it does to source reversal.

Two questions in one 2x2, plus a control.

**Does reversing the source still buy anything?** Chapter 5 measured +0.1789
exact match from Sutskever et al.'s trick, all of it on long sentences, and the
paper's own explanation was about distance: reversing puts the first source
words next to the first target words and shortens the minimal time lag. If that
explanation is right, then a model with a direct path from every source
position to every target position should not care, because there is no long
path left to shorten.

**Does the backward pass earn its keep?** Section 3.2 wants an annotation to
summarise "not only the preceding words, but also the following words". Drop
the backward RNN and the annotation only sees the prefix. Everything else about
the model stays: still one annotation per source position, still a weighted
sum, still equation (5).

The last row is the control that makes the second question answerable. A
bidirectional encoder at n_hidden 32 carries 40805 parameters and a
unidirectional one carries 27365, so the plain comparison confounds "reads both
ways" with "is bigger". A unidirectional encoder at n_hidden 42 carries 41495,
within two percent of the bidirectional model, and that is the row to read
against it.

Run: python experiments/ch06_encoder.py
"""

from __future__ import annotations

import time

from rnn_to_transformer_lab import attention
from rnn_to_transformer_lab.determinism import describe_environment, utf8_stdout
from rnn_to_transformer_lab.seq2seq import EPOCHS, N_TEST, N_TRAIN
from rnn_to_transformer_lab.toy_corpus import disjoint_splits, vocabularies

SEED = 0
SHORT = 11
#: (label, n_hidden, bidirectional, reverse_source). The first four are the
#: 2x2; the fifth is the parameter-matched control described above.
RUNS = (
    ("bidirectional", 32, True, False),
    ("bidirectional", 32, True, True),
    ("forward only", 32, False, False),
    ("forward only", 32, False, True),
    ("forward only", 42, False, False),
)


def main() -> None:
    utf8_stdout()
    started = time.perf_counter()
    print(describe_environment())

    source_vocab, target_vocab = vocabularies()
    train_pairs, test_pairs = disjoint_splits(N_TRAIN, N_TEST, seed=5)
    print(f"train {len(train_pairs)} test {len(test_pairs)} epochs {EPOCHS} "
          f"greedy decoding")
    print()

    print("encoder        d_h  source    params  loss     exact    short    long")
    for label, width, bidirectional, reverse in RUNS:
        model, losses = attention.train_one(
            SEED, reverse_source=reverse, n_hidden=width,
            train_pairs=train_pairs, bidirectional=bidirectional,
        )
        accuracy, hits = attention.greedy_accuracy(
            model, test_pairs, source_vocab, target_vocab,
            reverse_source=reverse,
        )
        short = [hit for length, hit in hits if length <= SHORT]
        long = [hit for length, hit in hits if length > SHORT]
        n_params = sum(p.numel() for p in model.parameters())
        print(
            f"{label:<14} {width:<4} {'reversed' if reverse else 'raw':<9} "
            f"{n_params:<7} {sum(losses[-20:]) / 20:<8.4f} {accuracy:<8.4f} "
            f"{sum(short) / len(short):<8.4f} {sum(long) / len(long):.4f}"
        )

    print()
    print(f"elapsed {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
