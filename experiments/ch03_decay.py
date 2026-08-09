"""Chapter 3: how fast the Jacobian product dies, against the spectral radius.

The well-behaved case. W_rec is normal, so its spectral radius and its spectral
norm are the same number and the paper's bound is tight. tanh gives gamma = 1,
so the vanishing condition reads radius < 1 with nothing else in it.

Run: python experiments/ch03_decay.py
"""

from __future__ import annotations

import time

import torch

from rnn_to_transformer_lab.determinism import describe_environment, seed_everything
from rnn_to_transformer_lab.jacobians import decay_rate, product_norms
from rnn_to_transformer_lab.rnn import GAMMA, PlainRNN, random_normal_matrix, spectral_norm, spectral_radius

N_HIDDEN = 64
N_STEPS = 100
RADII = (0.5, 0.9, 1.0, 1.2)
#: Four distances, not five. This table is quoted in the book, and a line wider
#: than about 70 characters wraps inside the measure there.
REPORT_AT = (1, 10, 50, 100)


def main() -> None:
    started = time.perf_counter()
    seed_everything(0)
    print(describe_environment())
    print(f"n_hidden={N_HIDDEN} n_steps={N_STEPS} activation=tanh gamma={GAMMA['tanh']}")
    print()

    generator = torch.Generator().manual_seed(0)
    x0 = torch.randn(N_HIDDEN, generator=generator, dtype=torch.float64) * 0.1

    header = "radius  norm   " + " ".join(f"{'l=' + str(l):<11}" for l in REPORT_AT) + "rate"
    print(header)
    for radius in RADII:
        gen = torch.Generator().manual_seed(1)
        w_hh = random_normal_matrix(N_HIDDEN, radius, gen)
        model = PlainRNN(
            w_hh=w_hh,
            w_xh=torch.zeros(N_HIDDEN, 1, dtype=torch.float64),
            b_h=torch.zeros(N_HIDDEN, dtype=torch.float64),
            act="tanh",
        )
        states = model.unroll(x0, N_STEPS)
        norms = product_norms(model, states, k=0)
        rate = decay_rate(norms, 50, 100)
        cells = " ".join(f"{norms[l]:<11.4e}" for l in REPORT_AT)
        print(f"{spectral_radius(w_hh):.3f}   {spectral_norm(w_hh):.3f}  {cells}{rate:.4f}")

    print()
    print(f"elapsed {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
