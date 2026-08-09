"""The single-hidden-unit error surface, and the wall in it.

Section 2.3 of Pascanu et al. reduces the whole problem to two parameters so
that the error surface can be drawn. Their equation (8) is

    x_t = w sigma(x_{t-1}) + b

with no input at all, and the cost is measured once, at the end:

    E_50 = (sigma(x_50) - 0.7)^2,   x_0 = 0.5,  sigma the logistic sigmoid.

Their figure 6 plots that surface over w near 5 and b near -2.4 and shows a
near-vertical wall. The claim the chapter needs is not that the picture looks
dramatic; it is that the gradient norm changes by orders of magnitude between
two points a hundredth apart, which is a number and can be measured.

Everything here runs in float64. The wall spans several orders of magnitude and
float32 loses the small side of it.
"""

from __future__ import annotations

import torch

#: The paper's constants for figure 6, kept in one place so that an experiment
#: cannot quietly drift from the setting the book says it reproduces.
X0 = 0.5
TARGET = 0.7
N_STEPS = 50


def final_cost(w: torch.Tensor, b: torch.Tensor, n_steps: int = N_STEPS) -> torch.Tensor:
    """E_50 for one (w, b), differentiable in both."""
    x = torch.as_tensor(X0, dtype=w.dtype)
    for _ in range(n_steps):
        x = w * torch.sigmoid(x) + b
    return (torch.sigmoid(x) - TARGET) ** 2


def cost_and_gradient(w_value: float, b_value: float, n_steps: int = N_STEPS) -> tuple[float, float, float]:
    """Return (cost, dE/dw, dE/db) at one point of the surface."""
    w = torch.tensor(w_value, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(b_value, dtype=torch.float64, requires_grad=True)
    cost = final_cost(w, b, n_steps)
    grad_w, grad_b = torch.autograd.grad(cost, (w, b))
    return cost.item(), grad_w.item(), grad_b.item()


def gradient_norm(w_value: float, b_value: float, n_steps: int = N_STEPS) -> float:
    """Euclidean norm of the gradient at one point."""
    _, grad_w, grad_b = cost_and_gradient(w_value, b_value, n_steps)
    return (grad_w**2 + grad_b**2) ** 0.5


def scan_b(
    w_value: float, b_start: float, b_stop: float, n_points: int, n_steps: int = N_STEPS
) -> list[tuple[float, float, float]]:
    """Walk b across the wall at fixed w.

    Returns (b, cost, gradient norm) per point. A scan along one axis is what
    makes the wall a measurement rather than a rendering: the cost climbs
    smoothly, and the gradient norm spikes over a handful of grid points.
    """
    step = (b_stop - b_start) / (n_points - 1)
    rows = []
    for i in range(n_points):
        b = b_start + i * step
        cost, grad_w, grad_b = cost_and_gradient(w_value, b, n_steps)
        rows.append((b, cost, (grad_w**2 + grad_b**2) ** 0.5))
    return rows
