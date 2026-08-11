"""Chapter 6: how far the gradient reaches back into the source, in both models.

The derivation says that in chapter 5's model the only route from a target step
to source word j runs along the encoder recurrence from j to T and then through
one fixed vector, so it is subject to exactly the decay chapter 3 derived; and
that equation (5) adds a route of length one, because c_i is a weighted sum in
which h_j appears directly.

This measures that. For one sentence and one target step, take the loss at that
step alone, backpropagate, and read the norm of the gradient arriving at each
source word's embedding. The profile is reported relative to its own largest
entry, because the two architectures have different overall scales and the
claim is about shape, not size. `min/max` is that shape in one number: 1.0 is a
perfectly flat reach, and small means the far end of the sentence is getting a
fraction of what the near end gets.

**Measured at initialization, before any training.** That is deliberate and it
is the same choice chapter 4 made for the carousel derivative. The claim under
test is a property of the computational graph - which paths exist and what they
do to a gradient - and training is what happens *because* of those paths. A
number measured after training would be a number about what this corpus taught
this model, which is a different question and belongs to the alignment script.

Note the alignment model starts with v_a exactly zero (appendix B.1), so every
score is zero on the first batch and the attention weights are uniform. The
flat profile below is therefore the honest starting point rather than a trained
model showing off.

Run: python experiments/ch06_gradient.py
"""

from __future__ import annotations

import time

import torch
from torch import nn

from rnn_to_transformer_lab import attention, seq2seq
from rnn_to_transformer_lab.determinism import describe_environment, utf8_stdout
from rnn_to_transformer_lab.seq2seq import N_TEST, N_TRAIN
from rnn_to_transformer_lab.toy_corpus import _pad, disjoint_splits, vocabularies

WIDTH = 32
SEED = 0
#: Which target step carries the loss. Step 0 is not representative: the
#: decoder's initial state is built from the encoder by a path of its own, so
#: the first step has an extra route that no later step has.
STEP = 5


def source_gradient(model, source, target, step: int) -> torch.Tensor:
    """Norm of dL_step / d(source embedding) at each source position."""
    captured: dict[str, torch.Tensor] = {}

    def keep(module, args, output):
        output.retain_grad()
        captured["embedded"] = output

    handle = model.source_embedding.register_forward_hook(keep)
    model.zero_grad(set_to_none=True)
    logits = model(source, target)
    nn.functional.cross_entropy(logits[step], target[1:][step]).backward()
    handle.remove()
    return captured["embedded"].grad[:, 0, :].norm(dim=-1)


def build(source_vocab, target_vocab):
    """One of each model, from the same seed so the embeddings start equal."""
    torch.manual_seed(SEED)
    fixed = seq2seq.Seq2Seq(source_vocab, target_vocab, n_hidden=WIDTH)
    torch.manual_seed(SEED)
    attn = attention.AttentionSeq2Seq(
        source_vocab, target_vocab, n_hidden=WIDTH
    )
    return fixed, attn


def encode_pair(pair, source_vocab, target_vocab):
    source = _pad([source_vocab.encode(pair.source)])
    target = _pad([
        [target_vocab.index["<sos>"]]
        + target_vocab.encode(pair.target)
        + [target_vocab.index["<eos>"]]
    ])
    return source, target


def main() -> None:
    utf8_stdout()
    started = time.perf_counter()
    print(describe_environment())

    source_vocab, target_vocab = vocabularies()
    _, test_pairs = disjoint_splits(N_TRAIN, N_TEST, seed=5)
    fixed, attn = build(source_vocab, target_vocab)
    print(f"d_hidden {WIDTH}, no training, loss taken at target step {STEP}")
    print()

    # One sentence per distinct source length, shortest first, so the table
    # shows what happens to the reach as the sentence grows.
    by_length: dict[int, object] = {}
    for pair in test_pairs:
        by_length.setdefault(len(pair.source), pair)
    lengths = sorted({*sorted(by_length)[::4], max(by_length)})

    print("source_len  fixed min/max  attention min/max  ratio")
    for length in lengths:
        source, target = encode_pair(
            by_length[length], source_vocab, target_vocab
        )
        if target.shape[0] - 1 <= STEP:
            continue
        one = source_gradient(fixed, source, target, STEP)
        two = source_gradient(attn, source, target, STEP)
        flat_one = float(one.min() / one.max())
        flat_two = float(two.min() / two.max())
        print(f"{length:<11} {flat_one:<14.6f} {flat_two:<18.6f} "
              f"{flat_two / flat_one:.1f}x")

    longest = by_length[max(by_length)]
    source, target = encode_pair(longest, source_vocab, target_vocab)
    one = source_gradient(fixed, source, target, STEP)
    two = source_gradient(attn, source, target, STEP)
    print()
    print(f"profile on the longest test sentence ({len(longest.source)} tokens),")
    print("each column divided by its own largest entry:")
    print()
    print("j   source word   fixed     attention")
    for j, word in enumerate(longest.source):
        print(f"{j + 1:<3} {word:<13} {one[j] / one.max():<9.4f} "
              f"{two[j] / two.max():.4f}")

    print()
    print(f"elapsed {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
