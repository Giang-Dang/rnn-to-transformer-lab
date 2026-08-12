"""Chapter 8: table 1's asymptotics turned into counted FLOPs and bytes.

Five tables, and nothing here trains or times anything, so every number is
reproducible on any machine to the last digit.

1. The analytic count of `cost.py` checked against `FlopCounterMode`, which is
   torch's own instrumentation of the operators that actually ran. If those two
   disagree the analytic formula is wrong, and every other table here is built
   on it.
2. Where the quadratic term sits inside one encoder layer, against n, at the
   paper's own base configuration.
3. The two crossovers in closed form, against the paper's "n smaller than d".
4. The score matrix in bytes, which is the resource that breaks first.
5. The paper's own "Training Cost (FLOPs)" column, reproduced from its section
   5.2 wall-clock figures and footnote 5. This is the one table here that is
   about the paper rather than about the architecture.

Run: python experiments/ch08_flops.py
"""

from __future__ import annotations

import time

import torch
from torch.utils.flop_counter import FlopCounterMode

from rnn_to_transformer_lab.cost import (
    attention_beats_lstm_below,
    encoder_layer_flops,
    lstm_layer_flops,
    quadratic_half_point,
    rnn_layer_flops,
    score_matrix_bytes,
)
from rnn_to_transformer_lab.determinism import (
    describe_environment,
    seed_everything,
    utf8_stdout,
)
from rnn_to_transformer_lab.transformer import EncoderLayer

#: Section 3.2.2 and 3.3: the paper's base model.
D_MODEL = 512
D_FF = 2048
N_HEADS = 8
N_LAYERS = 6

#: Section 5.2 and footnote 5 of section 6.1.
P100_TFLOPS = 9.5
N_GPUS = 8
BASE_HOURS = 12.0
BIG_DAYS = 3.5
PAPER_BASE_COST = 3.3e18
PAPER_BIG_COST = 2.3e19


def main() -> None:
    utf8_stdout()
    started = time.perf_counter()
    print(describe_environment())
    seed_everything(0)

    print()
    print("1. analytic count against torch's own FlopCounterMode")
    print(f"one EncoderLayer, d_model={D_MODEL}, d_ff={D_FF}, heads={N_HEADS}, batch 1")
    print()
    print("n      analytic        counted         difference")
    layer = EncoderLayer(D_MODEL, N_HEADS, D_FF)
    for n in (8, 32, 128, 512, 1024):
        analytic = encoder_layer_flops(n, D_MODEL, D_FF).total
        x = torch.randn(1, n, D_MODEL)
        with torch.no_grad():
            with FlopCounterMode(display=False) as counter:
                layer(x, None)
        counted = counter.get_total_flops()
        print(f"{n:<6} {analytic:<15,} {counted:<15,} {analytic - counted:,}")

    print()
    print("2. where the n^2 part sits inside one layer, base configuration")
    print("fraction depends only on the ratio n/d_model, never on n alone")
    print()
    print("n      projections     scores+values   ffn             n^2 share")
    for n in (16, 64, 256, 512, 1024, 2048, 3072, 4096, 8192):
        f = encoder_layer_flops(n, D_MODEL, D_FF)
        print(
            f"{n:<6} {f.projections:<15,} {f.quadratic:<15,} "
            f"{f.feed_forward:<15,} {f.quadratic_fraction:.4f}"
        )

    print()
    print("3. the crossovers, closed form")
    print()
    print("d_model  d_ff   half of a layer   attention vs LSTM   paper says")
    for d_model in (64, 128, 256, 512, 1024):
        d_ff = 4 * d_model
        print(
            f"{d_model:<8} {d_ff:<6} n = {quadratic_half_point(d_model, d_ff):<13} "
            f"n = {attention_beats_lstm_below(d_model):<15} n = {d_model}"
        )
    print()
    print("one layer's FLOPs at the three widths a recurrent layer can mean,")
    print(f"n = {D_MODEL} and d = {D_MODEL}:")
    n = D_MODEL
    attention_only = encoder_layer_flops(n, D_MODEL, D_FF)
    print(f"  self-attention sub-layer  {attention_only.projections + attention_only.quadratic:,}")
    print(f"  LSTM layer (4 gates)      {lstm_layer_flops(n, D_MODEL):,}")
    print(f"  plain recurrent layer     {rnn_layer_flops(n, D_MODEL):,}")

    print()
    print("4. the score tensor, in bytes")
    print(f"batch 8, {N_HEADS} heads, float32, one attention sub-layer")
    print()
    print("n      bytes               GiB        x a recurrent layer's (b,n,d)")
    for n in (128, 512, 1024, 2048, 4096, 8192):
        raw = score_matrix_bytes(n, N_HEADS, batch=8)
        recurrent = 8 * n * D_MODEL * 4
        print(
            f"{n:<6} {raw:<19,} {raw / 2**30:<10.4f} {raw / recurrent:.1f}"
        )

    print()
    print("5. the paper's own Training Cost (FLOPs) column, reproduced")
    print("section 6.1: time * GPUs * sustained single-precision capacity")
    print(f"footnote 5: {P100_TFLOPS} TFLOPS for a P100; section 5.2: {N_GPUS} GPUs")
    print()
    print("model  seconds     reproduced    paper prints  ratio")
    for name, seconds, printed in (
        ("base", BASE_HOURS * 3600, PAPER_BASE_COST),
        ("big", BIG_DAYS * 86400, PAPER_BIG_COST),
    ):
        rebuilt = seconds * N_GPUS * P100_TFLOPS * 1e12
        print(
            f"{name:<6} {seconds:<11,.0f} {rebuilt:<13.4e} {printed:<13.1e} "
            f"{rebuilt / printed:.4f}"
        )

    print()
    print(f"elapsed {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
