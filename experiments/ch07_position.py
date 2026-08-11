"""Chapter 7: what section 3.5 is for, and what its hypothesis actually says.

Four measurements.

1. **Why anything is needed.** Self-attention is a weighted sum over a set, so
   permuting the input permutes the output and changes nothing else. Measured
   as a residual rather than argued, because the failure is silent: a model
   with no positional information trains, drops its loss, and cannot represent
   word order.

2. **The linear-shift hypothesis.** The paper: "for any fixed offset k,
   PE_{pos+k} can be represented as a linear function of PE_pos". The map is
   one 2x2 rotation per frequency pair, and the content of the claim is that
   the rotation does not depend on pos. Both halves are checked here.

3. **A property the paper does not claim, which is stronger.** The inner
   product PE(pos) . PE(pos+k) depends only on k. It falls straight out of the
   sin/cos pairing - each pair contributes cos(k * omega_i), with pos
   cancelling - and it is the cleanest sense in which this encoding is about
   relative position.

4. **The wavelength range, against the sentence describing it.** The paper says
   the wavelengths "form a geometric progression from 2pi to 10000 . 2pi". The
   longest one the formula can actually produce is 2pi * 10000^((d-2)/d),
   which is short of that and by a different amount at every d_model.

Run: python experiments/ch07_position.py
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
from rnn_to_transformer_lab.transformer import (
    PE_BASE,
    MultiHeadAttention,
    positional_encoding,
)

OFFSETS = (1, 2, 5, 13, 40)
POSITIONS = (0, 1, 7, 30, 54)


def main() -> None:
    utf8_stdout()
    started = time.perf_counter()
    print(describe_environment())
    seed_everything(0)

    print()
    print("1. self-attention alone cannot tell two orderings apart")
    print("permute the input, permute the output back, compare with the original")
    print()
    print("d_model  heads  steps  max|difference|")
    for d_model, heads, steps in ((16, 4, 6), (32, 4, 12), (64, 8, 20)):
        layer = MultiHeadAttention(d_model, heads)
        x = torch.randn(1, steps, d_model)
        order = torch.randperm(steps)
        with torch.no_grad():
            straight, _ = layer(x, x, x)
            shuffled, _ = layer(x[:, order], x[:, order], x[:, order])
        gap = float((straight[:, order] - shuffled).abs().max())
        print(f"{d_model:<8} {heads:<6} {steps:<6} {gap:.3e}")

    print()
    print("2. PE(pos+k) = M_k PE(pos), and M_k does not depend on pos")
    print("M_k built analytically per pair: rotation by k*omega_i")
    print()
    d_model = 64
    table = positional_encoding(120, d_model)
    print("offset k  max|M_k PE(pos) - PE(pos+k)|  over positions")
    for k in OFFSETS:
        worst = 0.0
        for pair in range(0, d_model, 2):
            omega = 1.0 / math.pow(PE_BASE, pair / d_model)
            angle = k * omega
            rotation = torch.tensor(
                [[math.cos(angle), math.sin(angle)],
                 [-math.sin(angle), math.cos(angle)]]
            )
            for pos in POSITIONS:
                mapped = rotation @ table[pos, pair : pair + 2]
                gap = float((mapped - table[pos + k, pair : pair + 2]).abs().max())
                worst = max(worst, gap)
        print(f"{k:<9} {worst:.3e}                    {len(POSITIONS)} tested")

    print()
    print("3. PE(pos) . PE(pos+k) depends on k only, d_model 64")
    print()
    print("offset k  pos=0     pos=1     pos=7     pos=30    spread")
    for k in OFFSETS:
        dots = [float(table[pos] @ table[pos + k]) for pos in POSITIONS]
        print(
            f"{k:<9} {dots[0]:<9.5f} {dots[1]:<9.5f} {dots[2]:<9.5f} "
            f"{dots[3]:<9.5f} {max(dots) - min(dots):.2e}"
        )

    print()
    print("4. but it is not monotone in k, so it does not encode distance")
    print("the usual gloss on measurement 3 says the product falls off with")
    print("distance; it falls, then turns back up, and after that one value")
    print("of the product answers to several offsets")
    print()
    print("d_model  k where it first rises again  value at k=1  min over k<200")
    for width in (32, 64, 512):
        wide = positional_encoding(220, width)
        dots = [float(wide[0] @ wide[k]) for k in range(200)]
        rise = next(
            k + 1 for k in range(1, len(dots) - 1) if dots[k + 1] > dots[k] + 1e-5
        )
        print(
            f"{width:<8} {rise:<29} {dots[1]:<13.4f} "
            f"{min(dots[1:]):.4f} at k={dots.index(min(dots[1:]))}"
        )

    print()
    print("5. wavelengths the formula actually reaches, against 2pi to 10000*2pi")
    print()
    print("d_model  shortest/2pi  longest/2pi  paper says  shortfall")
    for width in (64, 128, 512):
        longest = math.pow(PE_BASE, (width - 2) / width)
        print(
            f"{width:<8} {1.0:<13.4f} {longest:<12.2f} {PE_BASE:<11.1f} "
            f"{PE_BASE / longest:.4f}x"
        )

    print()
    print(f"elapsed {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
