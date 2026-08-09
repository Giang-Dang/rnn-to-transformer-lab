"""Chapter 3: what the paper's regularizer is looking at, term by term.

Equation (9) penalises every step where the backward signal changes size. This
prints the ratios it is built from, before the squaring, for three spectral
radii, so the chapter can show what the term sees rather than only what it
costs.

Run: python experiments/ch03_regularizer.py
"""

from __future__ import annotations

import time

import torch

from rnn_to_transformer_lab.determinism import describe_environment, seed_everything
from rnn_to_transformer_lab.regularizer import omega, step_ratios
from rnn_to_transformer_lab.rnn import PlainRNN, random_normal_matrix, spectral_radius

N_HIDDEN = 32
N_STEPS = 30
RADII = (0.5, 0.9, 1.0)
REPORT_AT = (0, 5, 10, 20, 28)


def backward_signals(states: list[torch.Tensor]) -> list[torch.Tensor]:
    """dE/da_k for every k, for the cost E = 0.5 ||x_T||^2.

    A norm of the final state rather than a task loss: chapter 3 is about what
    the recurrence does to a signal travelling backwards, and any cost that
    depends only on x_T isolates that from the shape of a task.
    """
    cost = 0.5 * torch.sum(states[-1] ** 2)
    grads = torch.autograd.grad(cost, states, retain_graph=False, allow_unused=True)
    return [torch.zeros_like(states[i]) if g is None else g for i, g in enumerate(grads)]


def main() -> None:
    started = time.perf_counter()
    seed_everything(0)
    print(describe_environment())
    print(f"n_hidden={N_HIDDEN} n_steps={N_STEPS} activation=tanh cost=0.5*||x_T||^2")
    print()

    generator = torch.Generator().manual_seed(0)
    x0_base = torch.randn(N_HIDDEN, generator=generator, dtype=torch.float64) * 0.1

    print(f"{'radius':>7} " + " ".join(f"ratio@k={k:<4}" for k in REPORT_AT) + f" {'mean':>9} {'Omega':>12}")
    for radius in RADII:
        gen = torch.Generator().manual_seed(1)
        w_hh = random_normal_matrix(N_HIDDEN, radius, gen)
        model = PlainRNN(
            w_hh=w_hh,
            w_xh=torch.zeros(N_HIDDEN, 1, dtype=torch.float64),
            b_h=torch.zeros(N_HIDDEN, dtype=torch.float64),
            act="tanh",
        )
        x0 = x0_base.clone().requires_grad_(True)
        states = model.unroll(x0, N_STEPS)
        signals = backward_signals(states)
        ratios = step_ratios(model, states, signals)
        total = omega(model, states, signals)
        cells = " ".join(f"{ratios[k]:<11.6f}" for k in REPORT_AT)
        mean = sum(ratios) / len(ratios)
        print(f"{spectral_radius(w_hh):7.3f} {cells} {mean:9.6f} {total.item():12.6f}")

    print()
    print("A ratio of 1 is a step that changed nothing. Omega is the summed squared")
    print("distance from that, so a network whose gradient dies has a large Omega.")
    print()
    print(f"elapsed {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
