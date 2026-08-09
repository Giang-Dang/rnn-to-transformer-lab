"""Products of Jacobians through time, which is where the gradient goes.

Equation (5) of Pascanu et al. writes the factor that carries the error from
step t back to step k as a product of t - k Jacobians. Everything chapter 3
measures is a norm of that product, so it is computed once here and reused.

The paper's sentence about it is worth keeping in view: "In the same way a
product of t - k real numbers can shrink to zero or explode to infinity, so
does this product of matrices (along some direction v)." The parenthesis is
the part that a scalar intuition drops, and it is the part `transient_growth`
exists to show.
"""

from __future__ import annotations

import torch

from .rnn import PlainRNN


def jacobian_product(model: PlainRNN, states: list[torch.Tensor], k: int, t: int) -> torch.Tensor:
    """d x_t / d x_k, as the product of Jacobians from equation (5).

    `states` is what `PlainRNN.unroll` returned, so `states[i]` is the state at
    time i. The product runs over i = k+1 ... t of d x_i / d x_{i-1}, each
    evaluated at the state that step actually saw.
    """
    if not 0 <= k <= t < len(states):
        raise IndexError(f"need 0 <= k <= t < {len(states)}, got k={k}, t={t}")
    n = model.n_hidden
    product = torch.eye(n, dtype=states[0].dtype)
    for i in range(k + 1, t + 1):
        product = model.jacobian_at(states[i - 1]) @ product
    return product


def product_norms(model: PlainRNN, states: list[torch.Tensor], k: int = 0) -> list[float]:
    """Spectral norm of d x_t / d x_k for every t from k to the end.

    Returned list is indexed by distance l = t - k, so entry 0 is the norm of
    the identity and is always 1. Computed incrementally: recomputing the whole
    product for every t is quadratic and gives the same answer.
    """
    n = model.n_hidden
    product = torch.eye(n, dtype=states[0].dtype)
    norms = [torch.linalg.matrix_norm(product, ord=2).item()]
    for i in range(k + 1, len(states)):
        product = model.jacobian_at(states[i - 1]) @ product
        norms.append(torch.linalg.matrix_norm(product, ord=2).item())
    return norms


def linear_power_norms(w: torch.Tensor, max_power: int) -> list[float]:
    """Spectral norm of W^l for l = 0 ... max_power.

    The linear case, sigma set to the identity, which is where the paper's
    supplementary proof lives: there the Jacobian product collapses to a matrix
    power, equation (12). Nothing about the trajectory enters, so this isolates
    the linear algebra from the saturation of the nonlinearity.
    """
    n = w.shape[0]
    power = torch.eye(n, dtype=w.dtype)
    norms = [torch.linalg.matrix_norm(power, ord=2).item()]
    for _ in range(max_power):
        power = w @ power
        norms.append(torch.linalg.matrix_norm(power, ord=2).item())
    return norms


def transient_growth(norms: list[float]) -> tuple[int, float]:
    """Where a decaying sequence peaks, and how high.

    A spectral radius below 1 forces the product to zero eventually. It says
    nothing about what happens first. For a non-normal matrix the norm can
    climb for tens of steps before the decay takes over, and this returns the
    argmax and the max so an experiment can print both.
    """
    peak_index = max(range(len(norms)), key=norms.__getitem__)
    return peak_index, norms[peak_index]


def decay_rate(norms: list[float], start: int, stop: int) -> float:
    """Geometric decay factor per step, fitted over [start, stop].

    The paper's bound is eta^(t-k) for some eta < 1, so the natural summary of
    a measured curve is the per-step ratio implied by its two endpoints. Fitted
    over a window rather than from the first step, because the first steps are
    where the transient lives.
    """
    if not 0 <= start < stop < len(norms):
        raise IndexError(f"need 0 <= start < stop < {len(norms)}")
    if norms[start] <= 0.0 or norms[stop] <= 0.0:
        raise ValueError("cannot fit a geometric rate through a zero norm")
    return (norms[stop] / norms[start]) ** (1.0 / (stop - start))
