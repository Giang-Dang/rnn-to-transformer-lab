"""Chapter 7: why equation (1) divides by sqrt(d_k), measured rather than assumed.

Footnote 4 of the paper gives the reason in two sentences: if the components of
q and k are independent with mean 0 and variance 1, then q . k has mean 0 and
variance d_k. The main text adds what that costs - large dot products push "the
softmax function into regions where it has extremely small gradients".

Three tables here, and the third is the one the paper does not have.

1. The footnote's own claim, checked. Variance of q . k against d_k.
2. What the variance does to a softmax over 64 keys: how much mass the largest
   entry takes, the entropy, and the size of the softmax Jacobian, which is
   what "extremely small gradients" means as a number.
3. The assumption itself. The footnote says "mean 0 and variance 1", and a
   projection built with PyTorch's default initialisation does not produce
   that. The scaling is derived from a premise the code does not satisfy, and
   the third table says by how much it misses.

Run: python experiments/ch07_scaling.py
"""

from __future__ import annotations

import math
import time

import torch
from torch import nn

from rnn_to_transformer_lab.determinism import (
    describe_environment,
    seed_everything,
    utf8_stdout,
)

WIDTHS = (8, 16, 32, 64, 128, 256, 512, 1024)
#: Keys per attention row. 64 rather than a sentence length, so the softmax is
#: over enough entries for the entropy to have room to move.
N_KEYS = 64
N_SAMPLES = 20000


def jacobian_norm(p: torch.Tensor) -> float:
    """||diag(p) - p p^T||_F, the size of the softmax's own derivative.

    Zero when p is one-hot, largest when p is spread out. This is the quantity
    the paper's phrase "extremely small gradients" is about: whatever the loss
    downstream, every gradient reaching the scores is multiplied by this.
    """
    jacobian = torch.diag(p) - torch.outer(p, p)
    return float(torch.linalg.matrix_norm(jacobian))


def main() -> None:
    utf8_stdout()
    started = time.perf_counter()
    print(describe_environment())
    seed_everything(0)

    print()
    print("footnote 4: q.k has mean 0 and variance d_k when q,k ~ N(0,1)")
    print(f"{N_SAMPLES} samples per row")
    print()
    print("d_k    mean       var         var/d_k   sd        sqrt(d_k)")
    for d_k in WIDTHS:
        q = torch.randn(N_SAMPLES, d_k)
        k = torch.randn(N_SAMPLES, d_k)
        dots = (q * k).sum(dim=-1)
        var = float(dots.var())
        print(
            f"{d_k:<6} {float(dots.mean()):<10.4f} {var:<11.4f} "
            f"{var / d_k:<9.4f} {math.sqrt(var):<9.4f} {math.sqrt(d_k):.4f}"
        )

    print()
    print(f"what that does to one softmax over {N_KEYS} keys, averaged over 200 rows")
    print("grad is the Frobenius norm of the softmax Jacobian")
    print()
    print("       unscaled                    scaled by 1/sqrt(d_k)")
    print("d_k    max_p   entropy  grad       max_p   entropy  grad")
    for d_k in WIDTHS:
        columns = [[], [], [], [], [], []]
        for _ in range(200):
            q = torch.randn(d_k)
            k = torch.randn(N_KEYS, d_k)
            raw = k @ q
            for offset, scores in ((0, raw), (3, raw / math.sqrt(d_k))):
                p = torch.softmax(scores, dim=-1)
                entropy = float(-(p * torch.log(p + 1e-12)).sum())
                columns[offset].append(float(p.max()))
                columns[offset + 1].append(entropy)
                columns[offset + 2].append(jacobian_norm(p))
        cells = [sum(column) / len(column) for column in columns]
        print(
            f"{d_k:<6} {cells[0]:<7.4f} {cells[1]:<8.4f} {cells[2]:<10.6f} "
            f"{cells[3]:<7.4f} {cells[4]:<8.4f} {cells[5]:.6f}"
        )

    print()
    print("the premise: what a real projection actually produces")
    print("x ~ N(0,1); q = Linear(d_model, d_k)(x) at PyTorch's default init")
    print()
    print("d_model  d_k    var(component)  var(q.k)   d_k     ratio")
    for d_model, d_k in ((512, 64), (512, 512), (64, 16), (64, 64)):
        x = torch.randn(N_SAMPLES, d_model)
        to_query = nn.Linear(d_model, d_k, bias=False)
        to_key = nn.Linear(d_model, d_k, bias=False)
        with torch.no_grad():
            q, k = to_query(x), to_key(x)
        dots = (q * k).sum(dim=-1)
        component = float(q.var())
        var = float(dots.var())
        print(
            f"{d_model:<8} {d_k:<6} {component:<15.6f} {var:<10.6f} "
            f"{d_k:<7} {var / d_k:.6f}"
        )

    print()
    print(f"elapsed {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
