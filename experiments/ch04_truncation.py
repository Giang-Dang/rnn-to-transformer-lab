"""Chapter 4: what the paper's truncation actually costs.

The 1997 algorithm does not compute the gradient. It computes the gradient with
three derivatives replaced by zero, and the abstract says so in its second
sentence: "Truncating the gradient where this does not do harm". The claim that
it does no harm is the paper's, backed by "a few experiments with non-truncated
LSTM" that found no significant difference. It is worth a number.

Two quantities, over the same sequence and the same weights:

* the angle between the truncated gradient and the full one, as a cosine over
  all nine weight tensors flattened together. This is the one that matters for
  a descent direction: a cosine near 1 means the truncated step goes almost
  where the full step goes.
* the ratio of their norms, which says whether the truncation is also changing
  the step length.

Both are reported against the sequence length, because the paper's own reason
for expecting no harm is that outside the carousel the error vanishes quickly
anyway, and that reason gets stronger as the lag grows rather than weaker.

Run: python experiments/ch04_truncation.py
"""

from __future__ import annotations

import time

import torch

from rnn_to_transformer_lab.determinism import describe_environment, seed_everything
from rnn_to_transformer_lab.lstm import random_lstm

N_HIDDEN = 16
N_INPUT = 2
LENGTHS = (10, 25, 50, 100)
SCALE = 0.1
SEED = 3
#: One random sequence gives one cosine, and the spread across sequences is
#: wide enough that a single draw would not show a trend. Averaged instead.
N_DRAWS = 20


def gradient(
    model, inputs: torch.Tensor, readout: torch.Tensor, truncate: bool
) -> tuple[torch.Tensor, float]:
    """Flattened gradient of a terminal loss over all nine weight tensors."""
    states = model.unroll(inputs, truncate=truncate)
    prediction = readout @ states[-1].h
    loss = 0.5 * (prediction - 1.0) ** 2
    grads = torch.autograd.grad(loss, model.parameters())
    return torch.cat([g.reshape(-1) for g in grads]), loss.item()


def main() -> None:
    started = time.perf_counter()
    seed_everything(SEED)
    print(describe_environment())
    print(f"n_hidden={N_HIDDEN} n_input={N_INPUT} scale={SCALE} draws={N_DRAWS}")
    print()

    print("    T  mean cosine  min cosine   mean norm ratio")
    for length in LENGTHS:
        cosines: list[float] = []
        ratios: list[float] = []
        for draw in range(N_DRAWS):
            gen = torch.Generator().manual_seed(SEED + draw)
            model = random_lstm(
                N_HIDDEN, N_INPUT, gen, scale=SCALE, requires_grad=True
            )
            inputs = torch.randn(length, N_INPUT, generator=gen, dtype=torch.float64)
            readout = torch.randn(N_HIDDEN, generator=gen, dtype=torch.float64)
            truncated, loss_t = gradient(model, inputs, readout, truncate=True)
            full, loss_f = gradient(model, inputs, readout, truncate=False)
            assert abs(loss_t - loss_f) < 1e-12, "truncation moved the forward pass"
            cosines.append(
                torch.nn.functional.cosine_similarity(truncated, full, dim=0).item()
            )
            ratios.append((truncated.norm() / full.norm()).item())
        mean_cos = sum(cosines) / len(cosines)
        mean_ratio = sum(ratios) / len(ratios)
        print(
            f"{length:5d}  {mean_cos:<12.6f} {min(cosines):<12.6f} {mean_ratio:.6f}"
        )

    print()
    print(f"elapsed {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
