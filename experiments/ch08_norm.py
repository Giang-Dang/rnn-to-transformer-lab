"""Chapter 8: post-LN against pre-LN, at initialization and then trained.

Equation (5) of Vaswani et al. normalizes *after* the residual addition. Xiong
et al. (2020) prove that this choice is why the warmup schedule of section 5.3
has to exist, and their theorem 1 is a statement about initialization:

    post-LN:  || dL/dW^{2,L} ||_F <= O(d sqrt(ln d))
    pre-LN:   || dL/dW^{2,L} ||_F <= O(d sqrt(ln d / L))

The post-LN bound does not contain L. The pre-LN one shrinks with depth. Both
are upper bounds on the *last* layer's feed-forward matrix, which is what this
script measures, and both are proved in a reduced setting the chapter names:
single-head attention, and W^Q and W^K initialized to zero so that attention is
a uniform average. This model is none of those things - it is the repo's real
Transformer at its real initialization - so this is a test of whether the
theorem's shape survives contact with the model, not a check of its algebra.

Three tables.

1. The gradient at initialization against depth, both orders, no training.
2. What the two orders cost in parameters, which is not zero and is the only
   thing that can tell them apart from outside.
3. The two orders trained at this book's scale, with and without the paper's
   warmup schedule. This is where the theorem's practical claim either shows up
   or does not, and at two layers it is entitled not to.

Run: python experiments/ch08_norm.py
"""

from __future__ import annotations

import time

import torch

from rnn_to_transformer_lab.determinism import (
    describe_environment,
    seed_everything,
    utf8_stdout,
)
from rnn_to_transformer_lab.seq2seq import BATCH, N_TEST, N_TRAIN
from rnn_to_transformer_lab.toy_corpus import batches, disjoint_splits, vocabularies
from rnn_to_transformer_lab.transformer import (
    Transformer,
    greedy_accuracy,
    train_one,
    warmup_lambda,
)

D_MODEL = 24
N_HEADS = 4
DEPTHS = (1, 2, 4, 8, 16)
SEED = 0
#: Three seeds for the trained table. The gradient tables above need only one,
#: because nothing there is trained and the measurement is deterministic.
SEEDS = (0, 1, 2)
#: Enough steps for the schedule to have a shape inside this book's 658-step
#: budget. The paper's 4000 is more steps than a whole run here.
WARMUP_STEPS = 200


def gradient_norms(n_layers: int, norm_first: bool, source, target) -> dict[str, float]:
    """||grad||_F of the second FFN matrix at three places in the model.

    W^{2,l} in the theorem's notation is `feed_forward.outer.weight` here: the
    matrix that maps d_ff back down to d_model at layer l.

    The theorem is about "the last layer", meaning the one whose output the
    softmax reads. In an encoder-decoder that is the last *decoder* layer, not
    the last encoder layer - the encoder's output reaches the loss only through
    every decoder layer's cross-attention, so its gradient carries the whole
    decoder stack with it and is not the quantity theorem 1 bounds. Both are
    reported because the difference between them is worth seeing, but
    `decoder_last` is the one to read against the theorem.
    """
    seed_everything(SEED)
    source_vocab, target_vocab = vocabularies()
    model = Transformer(
        source_vocab, target_vocab, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=n_layers, norm_first=norm_first,
    )
    model.loss(source, target).backward()

    def norm(layer) -> float:
        return float(
            torch.linalg.matrix_norm(layer.feed_forward.outer.weight.grad)
        )

    return {
        "decoder_last": norm(model.decoder_layers[-1]),
        "decoder_first": norm(model.decoder_layers[0]),
        "encoder_last": norm(model.encoder_layers[-1]),
    }


def main() -> None:
    utf8_stdout()
    started = time.perf_counter()
    print(describe_environment())
    seed_everything(SEED)

    source_vocab, target_vocab = vocabularies()
    train_pairs, test_pairs = disjoint_splits(N_TRAIN, N_TEST, seed=5)
    generator = torch.Generator().manual_seed(100)
    one_batch = next(
        iter(batches(train_pairs, source_vocab, target_vocab, BATCH, generator))
    )

    print()
    print("1. the gradient at initialization, against depth")
    print(f"d_model = {D_MODEL}, {N_HEADS} heads, one batch of {BATCH}, no training")
    print("W^2,L is the last decoder layer's d_ff -> d_model matrix, the one")
    print("whose output the softmax reads; the encoder column is there for contrast")
    print()
    print("L    post-LN       pre-LN        post/pre   post enc last  pre enc last")
    measured = {}
    for depth in DEPTHS:
        post = gradient_norms(depth, False, *one_batch)
        pre = gradient_norms(depth, True, *one_batch)
        measured[depth] = (post, pre)
        print(
            f"{depth:<4} {post['decoder_last']:<13.6f} {pre['decoder_last']:<13.6f} "
            f"{post['decoder_last'] / pre['decoder_last']:<10.2f} "
            f"{post['encoder_last']:<14.6f} {pre['encoder_last']:.6f}"
        )
    print()
    print("the theorem says the post-LN column does not move with L and the")
    print("pre-LN column falls as 1/sqrt(L); these are the same numbers, divided")
    print("through by their own L=1 row so the shape is readable")
    print()
    print("L    post/post(L=1)  pre/pre(L=1)  1/sqrt(L)")
    base_post = measured[DEPTHS[0]][0]["decoder_last"]
    base_pre = measured[DEPTHS[0]][1]["decoder_last"]
    for depth in DEPTHS:
        post, pre = measured[depth]
        print(
            f"{depth:<4} {post['decoder_last'] / base_post:<15.4f} "
            f"{pre['decoder_last'] / base_pre:<13.4f} {depth ** -0.5:.4f}"
        )

    print()
    print("2. what the final LayerNorm costs")
    print("pre-LN needs one more norm per stack, so the counts are not equal")
    print()
    print("d_model  post-LN params  pre-LN params  difference  4*d_model")
    for d_model in (24, 64, 512):
        counts = []
        for norm_first in (False, True):
            seed_everything(SEED)
            model = Transformer(
                source_vocab, target_vocab, d_model=d_model, n_heads=N_HEADS,
                n_layers=2, norm_first=norm_first,
            )
            counts.append(sum(p.numel() for p in model.parameters()))
        print(
            f"{d_model:<8} {counts[0]:<15,} {counts[1]:<14,} "
            f"{counts[1] - counts[0]:<11} {4 * d_model}"
        )

    print()
    print("3. trained at this book's scale, 2 layers, 14 epochs")
    print("shared recipe = Adam at a flat 0.005; warmup = the paper's equation (3)")
    print(f"warmup_steps = {WARMUP_STEPS} against the paper's 4000, on 658 steps")
    print(f"{len(SEEDS)} seeds per row, because exact match on 300 sentences is a")
    print("noisy statistic and a four-cell comparison off one seed is not one")
    print()
    print("order    schedule  loss      exact     min      max      short   long")
    for norm_first in (False, True):
        for schedule_name in ("flat", "warmup"):
            schedule = (
                None if schedule_name == "flat"
                else warmup_lambda(D_MODEL, WARMUP_STEPS)
            )
            scores, shorts, longs, final_losses = [], [], [], []
            for seed in SEEDS:
                model, losses = train_one(
                    seed, reverse_source=True, d_model=D_MODEL, n_heads=N_HEADS,
                    n_layers=2, norm_first=norm_first,
                    learning_rate=None if schedule is None else 1.0,
                    schedule=schedule,
                )
                exact, hits = greedy_accuracy(
                    model, test_pairs, source_vocab, target_vocab,
                    reverse_source=True,
                )
                short = [hit for length, hit in hits if length <= 11]
                long = [hit for length, hit in hits if length > 11]
                scores.append(exact)
                shorts.append(sum(short) / len(short))
                longs.append(sum(long) / len(long))
                final_losses.append(sum(losses[-20:]) / 20)
            mean = lambda xs: sum(xs) / len(xs)
            print(
                f"{'pre-LN' if norm_first else 'post-LN':<8} "
                f"{schedule_name:<9} {mean(final_losses):<9.4f} "
                f"{mean(scores):<9.4f} {min(scores):<8.4f} {max(scores):<8.4f} "
                f"{mean(shorts):<7.4f} {mean(longs):.4f}"
            )
    print()
    print(f"seed {SEEDS[0]} alone, post-LN flat, is chapter 7's own row: it must")
    print("reproduce 0.1159 / 0.4700 / 0.8581 / 0.0921 or something here has moved")

    print()
    print(f"elapsed {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
