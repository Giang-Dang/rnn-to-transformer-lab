"""Chapter 6: what the alignment weights actually line up with.

Bahdanau et al. read their figure 3 by eye and report that the alignment is
"largely monotonic" with non-trivial excursions where French and English order
adjectives and nouns differently. This corpus is built so that the excursion is
the *whole* task rather than an occasional case: English writes "black cat" and
Vietnamese writes "meo den", so every adjective in the corpus has to cross its
noun.

That makes the qualitative claim checkable instead of impressionistic. For
every noun-adjective pair the model produces, take the source position each of
the two attended to most, and ask whether the adjective's is to the *left* of
the noun's. A diagonal alignment scores 0 here by construction. The count is
the crossing rate below.

The model is trained on unreversed source. Chapter 5 reversed it, and the
encoder script next door is where that choice is measured rather than assumed;
here it would only make the matrix read backwards.

Run: python experiments/ch06_alignment.py
"""

from __future__ import annotations

import time

import torch

from rnn_to_transformer_lab import attention
from rnn_to_transformer_lab.determinism import describe_environment, utf8_stdout
from rnn_to_transformer_lab.seq2seq import EPOCHS, N_TEST, N_TRAIN
from rnn_to_transformer_lab.toy_corpus import (
    ANIMAL_ADJECTIVES,
    ANIMALS,
    OBJECT_ADJECTIVES,
    OBJECTS,
    _pad,
    disjoint_splits,
    vocabularies,
)

WIDTH = 32
SEED = 0
#: The sentence the chapter prints as a matrix. Chosen for shape rather than
#: for result: one clause, one adjective on each noun phrase, short enough that
#: the matrix fits the page. The crossing rate below is over the whole test
#: split, so nothing rests on this one row.
SHOWN = ("the", "black", "cat", "wants", "a", "new", "lamp")

VI_ADJECTIVES = {vi: en for en, vi in OBJECT_ADJECTIVES + ANIMAL_ADJECTIVES}
VI_NOUNS = {vi: en for en, vi, _ in ANIMALS + OBJECTS}


def alignment(model, tokens, source_vocab):
    """Decode one sentence; return its output tokens and the alpha matrix."""
    source = _pad([source_vocab.encode(tokens)])
    rows, weights = attention.greedy_decode(
        model, source, keep_weights=True
    )
    matrix = torch.stack([w[:, 0] for w in weights])
    return rows[0], matrix


def crossings(model, pairs, source_vocab, target_vocab):
    """Count noun-adjective pairs whose attention crosses, over `pairs`.

    Walks the model's own output rather than the reference, so a sentence the
    model got wrong still contributes if it produced a noun followed by an
    adjective. What is being measured is where the model looked, not whether it
    was right.
    """
    crossed = total = 0
    for pair in pairs:
        out, matrix = alignment(model, pair.source, source_vocab)
        words = target_vocab.decode(out)
        peak = matrix.argmax(dim=1).tolist()
        for i in range(len(words) - 1):
            if words[i] in VI_NOUNS and words[i + 1] in VI_ADJECTIVES:
                total += 1
                crossed += peak[i + 1] < peak[i]
    return crossed, total


def main() -> None:
    utf8_stdout()
    started = time.perf_counter()
    print(describe_environment())

    source_vocab, target_vocab = vocabularies()
    train_pairs, test_pairs = disjoint_splits(N_TRAIN, N_TEST, seed=5)
    model, losses = attention.train_one(
        SEED, reverse_source=False, n_hidden=WIDTH, train_pairs=train_pairs
    )
    accuracy, _ = attention.greedy_accuracy(
        model, test_pairs, source_vocab, target_vocab
    )
    print(f"d_hidden {WIDTH}, epochs {EPOCHS}, source not reversed")
    print(f"loss {sum(losses[-20:]) / 20:.4f}  exact {accuracy:.4f}")
    print()

    out, matrix = alignment(model, SHOWN, source_vocab)
    words = target_vocab.decode(out)
    print("source: " + "  ".join(f"{j + 1}.{w}" for j, w in enumerate(SHOWN)))
    print()
    header = "  ".join(f"{j + 1:>4}" for j in range(len(SHOWN)))
    print(f"{'target':<10} {header}")
    for i, word in enumerate(words):
        row = "  ".join(f"{v:4.2f}" for v in matrix[i, : len(SHOWN)].tolist())
        print(f"{word:<10} {row}")

    print()
    crossed, total = crossings(model, test_pairs, source_vocab, target_vocab)
    print(f"noun-adjective pairs in the model's own output: {total}")
    print(f"of those, adjective attends left of the noun: {crossed}")
    print(f"crossing rate: {crossed / total:.4f}")
    print()
    print(f"elapsed {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
