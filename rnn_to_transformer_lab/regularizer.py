"""The paper's other remedy: a penalty that asks the error signal to keep its norm.

Equation (9) of Pascanu et al.:

    Omega = sum_k Omega_k
          = sum_k ( ||(dE/dx_{k+1}) (dx_{k+1}/dx_k)|| / ||dE/dx_{k+1}|| - 1 )^2

Read it as a ratio. The numerator is the error signal after one more step back
in time, the denominator is the same signal before that step, so the ratio is
what that step did to its size. Ask for a ratio of 1 at every step and you have
asked the backward pass to be norm-preserving. The square makes shrinking and
growing equally expensive.

The cost is in the shapes. Every term needs the backward signal dE/dx_{k+1} at
every step, which the ordinary backward pass computes and throws away, and then
a second derivative to push Omega itself back into W_rec. The paper takes the
documented shortcut: only the immediate derivative, with x_k and dE/dx_{k+1}
held constant. `detach` below is that shortcut, and it is the reason this term
is affordable at all.

Chapter 3 keeps this module to show what the losing remedy actually was.
Clipping is four lines and needs nothing the backward pass did not already
have; this needs the per-step backward signals and a second pass. Both worked
in the paper. Only one of them was cheap enough that everyone kept paying for
it.
"""

from __future__ import annotations

import torch

from .rnn import PlainRNN


def omega_terms(
    model: PlainRNN, states: list[torch.Tensor], backward_signals: list[torch.Tensor]
) -> list[torch.Tensor]:
    """One Omega_k per step, from equation (9).

    `backward_signals[k]` is dE/dx_k as a vector. Indices follow the paper:
    Omega_k compares the signal at k+1 with the same signal pushed one step
    further back, through dx_{k+1}/dx_k.

    Both the state and the incoming signal are detached, which is the paper's
    "immediate" derivative. Without it the term differentiates through the
    entire unrolled graph twice and stops being usable.
    """
    terms = []
    for k in range(len(backward_signals) - 1):
        signal = backward_signals[k + 1].detach()
        denominator = torch.linalg.vector_norm(signal)
        if denominator.item() == 0.0:
            continue
        jacobian = model.jacobian_at(states[k].detach())
        pushed = signal @ jacobian
        ratio = torch.linalg.vector_norm(pushed) / denominator
        terms.append((ratio - 1.0) ** 2)
    return terms


def omega(
    model: PlainRNN, states: list[torch.Tensor], backward_signals: list[torch.Tensor]
) -> torch.Tensor:
    """The summed penalty."""
    terms = omega_terms(model, states, backward_signals)
    if not terms:
        return torch.zeros((), dtype=states[0].dtype)
    return torch.stack(terms).sum()


def step_ratios(
    model: PlainRNN, states: list[torch.Tensor], backward_signals: list[torch.Tensor]
) -> list[float]:
    """The per-step ratio the penalty is built on, before it is squared.

    Printing these is how the chapter shows what the term is asking for: a
    column of numbers that should all be 1 and, in an untrained network with a
    small spectral radius, are all well under it.
    """
    ratios = []
    for k in range(len(backward_signals) - 1):
        signal = backward_signals[k + 1].detach()
        denominator = torch.linalg.vector_norm(signal)
        if denominator.item() == 0.0:
            continue
        pushed = signal @ model.jacobian_at(states[k].detach())
        ratios.append((torch.linalg.vector_norm(pushed) / denominator).item())
    return ratios
