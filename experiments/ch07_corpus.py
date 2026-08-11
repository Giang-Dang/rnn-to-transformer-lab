"""Chapter 7: the Transformer against chapter 6's model, on chapter 5's recipe.

Everything that could differ between this table and chapter 6's is taken from
`seq2seq` unchanged: the split, the batch size, the learning rate, the
optimizer, the source reversal, greedy decoding and exact-match scoring. Two
things differ, and the table has a column for each: the architecture, and the
number of epochs.

The epoch column is the point. Chapter 6's recipe stops at 14 epochs, and at 14
epochs this architecture loses badly. It is not at a ceiling - it is short of
steps, and the same run carried to 56 epochs arrives where chapter 6 arrived.
Reporting only one of those two numbers would be reporting half a result, so
the Transformer row is one training evaluated at four points along its own
trajectory rather than four separate runs. Same weights, same batch order,
same optimizer state: the row at 14 epochs is exactly the model a 14-epoch run
produces.

The first row is chapter 6's model retrained here rather than quoted, for the
reason chapter 6 gives about chapter 5: if it does not reproduce, every
difference further down the table could be a difference in the environment.

Sizes are chosen so the comparison survives the obvious objection. The
Transformer at d_model 24 carries 35845 parameters against the attention
model's 40805, so it is the smaller of the two and a win cannot be attributed
to capacity.

Run: python experiments/ch07_corpus.py
"""

from __future__ import annotations

import time

import torch

from rnn_to_transformer_lab import attention, transformer
from rnn_to_transformer_lab.determinism import describe_environment, utf8_stdout
from rnn_to_transformer_lab.seq2seq import (
    BATCH,
    EPOCHS,
    LEARNING_RATE,
    N_TEST,
    N_TRAIN,
)
from rnn_to_transformer_lab.toy_corpus import (
    batches,
    disjoint_splits,
    vocabularies,
)

#: Chapter 5's clause boundary, repeated rather than re-derived.
SHORT = 11
#: Where the Transformer row is scored. The first is chapter 6's own number, so
#: the two architectures are compared at the recipe chapter 6 actually ran.
CHECKPOINTS = (14, 28, 42, 56)
SEED = 0
#: 35845 parameters against the attention model's 40805, so the Transformer is
#: the smaller model in every row of this table.
D_MODEL = 24


def score(module, model, test_pairs, source_vocab, target_vocab):
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
    print(
        f"train {len(train_pairs)} test {len(test_pairs)} batch {BATCH} "
        f"lr {LEARNING_RATE}, source reversed, greedy decoding"
    )
    print()
    print("model        d     epochs  params   loss     exact    short    long")

    # Chapter 6's model at the recipe chapter 6 ran, reproduced rather than quoted.
    model, losses = attention.train_one(
        SEED, reverse_source=True, n_hidden=32, train_pairs=train_pairs
    )
    exact, short, long = score(
        attention, model, test_pairs, source_vocab, target_vocab
    )
    print(
        f"{'attention':<12} {32:<5} {EPOCHS:<7} "
        f"{sum(p.numel() for p in model.parameters()):<8} "
        f"{sum(losses[-20:]) / 20:<8.4f} {exact:<8.4f} {short:<8.4f} {long:.4f}"
    )

    # One Transformer, scored four times along its own trajectory.
    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(100 + SEED)
    batched = batches(
        train_pairs, source_vocab, target_vocab, BATCH, generator,
        reverse_source=True,
    )
    net = transformer.Transformer(
        source_vocab, target_vocab, d_model=D_MODEL, n_heads=4, n_layers=2
    )
    n_params = sum(p.numel() for p in net.parameters())
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    losses = []
    for epoch in range(1, max(CHECKPOINTS) + 1):
        for source, target in batched:
            loss = net.loss(source, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            optimizer.step()
            losses.append(loss.item())
        if epoch in CHECKPOINTS:
            exact, short, long = score(
                transformer, net, test_pairs, source_vocab, target_vocab
            )
            print(
                f"{'transformer':<12} {D_MODEL:<5} {epoch:<7} {n_params:<8} "
                f"{sum(losses[-20:]) / 20:<8.4f} {exact:<8.4f} "
                f"{short:<8.4f} {long:.4f}"
            )

    print()
    n_short = sum(1 for pair in test_pairs if len(pair.source) <= SHORT)
    print(
        f"short = source of at most {SHORT} tokens ({n_short} of "
        f"{len(test_pairs)}), long = the rest"
    )
    print(f"elapsed {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
