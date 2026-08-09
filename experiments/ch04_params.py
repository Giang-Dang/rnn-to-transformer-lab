"""Chapter 4: what the memory cell costs in parameters.

The number usually quoted is four: an LSTM layer has four times the weights of
a plain recurrent layer of the same width. The 1997 paper quotes a different
number, a factor of 3^2 in the fully connected case. Both are right, and they
are not two answers to one question.

The factor of four counts the 2000 cell in layer form: four weight blocks
(cell input, input gate, forget gate, output gate) against the plain layer's
one, each block reading the same h_{t-1}. The 1997 cell has three blocks, not
four, because it has no forget gate.

The factor of nine counts the paper's own topology, where the hidden layer is
fully connected and a gate receives connections from every memory cell and
every gate unit. Replacing one hidden unit by three units multiplies both the
number of sources and the number of destinations, so the weights go up by 3^2
rather than by 3.

Run: python experiments/ch04_params.py
"""

from __future__ import annotations

import time

import torch

from rnn_to_transformer_lab.determinism import describe_environment, seed_everything
from rnn_to_transformer_lab.lstm import (
    fully_connected_parameters,
    gru_parameters,
    layer_parameters,
    random_lstm,
)

D_HIDDEN = 256
D_INPUT = 256
#: The paper's own topology at a width small enough to be a real experiment of
#: its era, so the 3^2 is read off a count rather than from the exponent.
FC_UNITS = 64
FC_INPUT = 8


def main() -> None:
    started = time.perf_counter()
    seed_everything(0)
    print(describe_environment())
    print(f"layer form: d_hidden={D_HIDDEN} d_input={D_INPUT}")
    print()

    plain = layer_parameters(D_HIDDEN, D_INPUT, blocks=1)
    rows = [
        ("plain RNN", plain, 1),
        ("LSTM 1997", layer_parameters(D_HIDDEN, D_INPUT, blocks=3), 3),
        ("LSTM 2000", layer_parameters(D_HIDDEN, D_INPUT, blocks=4), 4),
        ("GRU 2014", gru_parameters(D_HIDDEN, D_INPUT), 3),
    ]
    print("model       blocks  parameters  vs plain RNN")
    for label, count, blocks in rows:
        print(f"{label:<11} {blocks:<7} {count:<11} {count / plain:.2f}")

    print()
    print(f"paper's topology: hidden layer fully connected, d_input={FC_INPUT}")
    one = fully_connected_parameters(FC_UNITS, FC_INPUT)
    three = fully_connected_parameters(3 * FC_UNITS, FC_INPUT)
    print(f"{FC_UNITS} plain units         {one}")
    print(f"{3 * FC_UNITS} units (3 per cell)  {three}")
    print(f"ratio, all weights     {three / one:.4f}")
    print(f"ratio, recurrent only  {(3 * FC_UNITS) ** 2 / FC_UNITS**2:.4f}")
    print("The paper's 3^2 is the recurrent block, where tripling the units")
    print("triples sources and destinations both. Input weights and biases")
    print("scale by 3 rather than 9, so the whole layer lands below 9.")

    print()
    counted = sum(p.numel() for p in random_lstm(D_HIDDEN, D_INPUT, torch.Generator()).parameters())
    print(f"counted from this repo's 1997 cell: {counted}")
    reference = torch.nn.LSTM(D_INPUT, D_HIDDEN, num_layers=1, bias=True)
    torch_count = sum(p.numel() for p in reference.parameters())
    print(f"counted from torch.nn.LSTM:         {torch_count}")
    print(f"torch.nn.LSTM vs plain RNN:         {torch_count / plain:.4f}")
    print("torch.nn.LSTM carries two bias vectors per block, not one,")
    print("which is why the ratio is above 4 rather than exactly 4.")

    print()
    print(f"elapsed {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
