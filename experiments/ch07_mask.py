"""Chapter 7: the decoder's mask, tested by intervention rather than by reading it.

Section 3.2.3: the mask goes "inside of scaled dot-product attention by masking
out (setting to -inf) all values in the input of the softmax which correspond
to illegal connections", and that, "combined with fact that the output
embeddings are offset by one position, ensures that the predictions for
position i can depend only on the known outputs at positions less than i".

Three measurements.

1. **Does it hold?** Change the target token at position j and see which output
   positions move. Everything before j must be bit-identical, and something at
   or after j must move, or the test would pass on a model that ignores its
   input.

2. **What it costs to get wrong.** The same intervention with the mask removed.
   This is the failure the chapter warns about, and the warning needs a number:
   without the mask the model reads the answer it is being asked to predict,
   the training loss falls faster, and nothing anywhere reports a problem.

3. **Inside the softmax against after it.** The paper says inside. On ordinary
   scores the two give the same answer to floating point, because restricting
   a softmax to a subset and renormalising is that subset's softmax - the
   normaliser cancels. So the reason to do it inside is not the mathematics. It
   is the row where the kept scores sit far below the masked ones: the kept
   mass underflows to zero before anything renormalises it, and the result is
   NaN. This measures both halves.

Run: python experiments/ch07_mask.py
"""

from __future__ import annotations

import math
import time

import torch

from rnn_to_transformer_lab.determinism import (
    describe_environment,
    seed_everything,
    utf8_stdout,
)
from rnn_to_transformer_lab.seq2seq import BATCH, EPOCHS, N_TEST, N_TRAIN
from rnn_to_transformer_lab.toy_corpus import (
    _pad,
    batches,
    disjoint_splits,
    vocabularies,
)
from rnn_to_transformer_lab.transformer import (
    Transformer,
    causal_mask,
    scaled_dot_product,
)


def intervene(model, source, column, cut, masked: bool):
    """Move the target token at `cut`; report how far each output shifted."""
    altered = column.clone()
    altered[cut, 0] = (altered[cut, 0] + 3) % len(model.target_vocab)
    with torch.no_grad():
        memory, mask = model.encode(source)
        if masked:
            straight, _ = model.decode(memory, mask, column)
            changed, _ = model.decode(memory, mask, altered)
        else:
            straight = _decode_unmasked(model, memory, mask, column)
            changed = _decode_unmasked(model, memory, mask, altered)
    return (straight - changed).abs().amax(dim=(1, 2))


def _decode_unmasked(model, memory, source_mask, target_in):
    """The same decoder with the causal mask removed. Nothing else changes."""
    tokens = target_in.transpose(0, 1)
    x = model._embed(tokens, model.target_embedding)
    for layer in model.decoder_layers:
        x, _ = layer(x, memory, None, source_mask)
    return model.readout(x).transpose(0, 1)


def main() -> None:
    utf8_stdout()
    started = time.perf_counter()
    print(describe_environment())
    seed_everything(0)

    source_vocab, target_vocab = vocabularies()
    torch.manual_seed(0)
    model = Transformer(source_vocab, target_vocab, d_model=32, n_heads=4)
    generator = torch.Generator().manual_seed(11)

    from rnn_to_transformer_lab.toy_corpus import sentence

    pair = sentence(generator, max_clauses=1)
    source = _pad([source_vocab.encode(pair.source)])
    tokens = [target_vocab.index["<sos>"]] + target_vocab.encode(pair.target)
    column = torch.tensor(tokens, dtype=torch.long).unsqueeze(1)
    cut = len(tokens) // 2

    print()
    print(f"target of {len(tokens)} positions; token at position {cut} changed")
    print("max|change| in the logits at each output position")
    print()
    print("position  with mask       without mask")
    for position in range(len(tokens)):
        with_mask = intervene(model, source, column, cut, masked=True)
        without = intervene(model, source, column, cut, masked=False)
        flag = "   <- changed" if position < cut else ""
        print(
            f"{position:<9} {float(with_mask[position]):<15.8f} "
            f"{float(without[position]):.8f}{flag}"
        )
        if position >= cut + 1:
            break

    print()
    print("what the leak costs, on the shared recipe: same seed, same batches")
    print("the unmasked decoder is scored the only way a decoder can be scored")
    print("at generation time, where there is no future to read")
    train_pairs, test_pairs = disjoint_splits(N_TRAIN, N_TEST, seed=5)
    print()
    print(f"decoder self-attention  train loss  exact match ({len(test_pairs)} test)")
    for label, masked in (("masked (the paper)", True), ("unmasked", False)):
        torch.manual_seed(0)
        gen = torch.Generator().manual_seed(100)
        batched = batches(
            train_pairs, source_vocab, target_vocab, BATCH, gen, reverse_source=True
        )
        trained = Transformer(source_vocab, target_vocab, d_model=32, n_heads=4)
        optimizer = torch.optim.Adam(trained.parameters(), lr=0.005)
        losses: list[float] = []
        for _ in range(EPOCHS):
            for src, tgt in batched:
                if masked:
                    logits = trained(src, tgt)
                else:
                    memory, mask = trained.encode(src)
                    logits = _decode_unmasked(trained, memory, mask, tgt[:-1])
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    tgt[1:].reshape(-1),
                    ignore_index=0,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trained.parameters(), 5.0)
                optimizer.step()
                losses.append(loss.item())
        from rnn_to_transformer_lab.transformer import greedy_accuracy

        exact, _ = greedy_accuracy(
            trained, test_pairs, source_vocab, target_vocab, reverse_source=True
        )
        print(f"{label:<23} {sum(losses[-20:]) / 20:<11.4f} {exact:.4f}")

    print()
    print("inside the softmax against after it, on the same scores")
    print()
    torch.manual_seed(3)
    query = torch.randn(1, 1, 6, 8)
    key = torch.randn(1, 1, 6, 8)
    value = torch.randn(1, 1, 6, 8)
    mask = causal_mask(6)
    _, inside = scaled_dot_product(query, key, value, mask)
    outside = torch.softmax(query @ key.transpose(-2, -1) / math.sqrt(8), dim=-1)
    outside = outside * mask
    outside = outside / outside.sum(dim=-1, keepdim=True)
    print(f"ordinary scores, max|inside - after| = {float((inside - outside).abs().max()):.3e}")

    scores = torch.tensor([[[[900.0, -900.0]]]])
    keep = torch.tensor([[[[False, True]]]])
    inside_row = torch.softmax(scores.masked_fill(~keep, float("-inf")), dim=-1)
    after_row = torch.softmax(scores, dim=-1) * keep
    after_row = after_row / after_row.sum(dim=-1, keepdim=True)
    print(f"scores [900, -900], keep the second only")
    print(f"  inside the softmax: {inside_row.flatten().tolist()}")
    print(f"  after it:           {after_row.flatten().tolist()}")

    print()
    print(f"elapsed {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
