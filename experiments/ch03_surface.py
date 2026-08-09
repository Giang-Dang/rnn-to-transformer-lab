"""Chapter 3: the wall in the error surface, measured rather than drawn.

Reproduces the setting of figure 6 in Pascanu et al.: one hidden unit, no
input, x_t = w sigmoid(x_{t-1}) + b, x_0 = 0.5, and the cost read once at step
50 as (sigmoid(x_50) - 0.7)^2.

The figure shows a wall. What a figure cannot show is how narrow it is, and
that is the number the chapter needs: the gradient norm at the top of the wall
against the gradient norm a few tenths away on the flat.

Run: python experiments/ch03_surface.py
"""

from __future__ import annotations

import math
import time

from rnn_to_transformer_lab.determinism import describe_environment
from rnn_to_transformer_lab.surface import N_STEPS, TARGET, X0, cost_and_gradient

#: The window figure 6 plots.
W_VALUES = (4.6, 4.8, 5.0, 5.2, 5.4)
B_LOW, B_HIGH = -2.8, -2.0
#: Widened for the peak search only: at w = 5.4 the wall has moved just past
#: the left edge of the window the paper drew.
SEARCH_LOW, SEARCH_HIGH = -3.0, -2.0
FLAT_B = -2.0
W_MAIN = 5.0


def gradient_norm_at(w: float, b: float) -> tuple[float, float]:
    cost, grad_w, grad_b = cost_and_gradient(w, b)
    return cost, math.hypot(grad_w, grad_b)


def refine_peak(w: float, low: float, high: float, points: int, rounds: int) -> tuple[float, float, float]:
    """Nested grid search for the steepest point along b at fixed w.

    Each round keeps the best point and shrinks the window by a factor of 50
    around it. Nested grids rather than a derivative search on purpose: the
    quantity being maximised is itself a gradient norm, and the surface it sits
    on is the thing under investigation.
    """
    best = (0.0, low, 0.0)
    for _ in range(rounds):
        step = (high - low) / (points - 1)
        best = (0.0, low, 0.0)
        for i in range(points):
            b = low + i * step
            cost, norm = gradient_norm_at(w, b)
            if norm > best[0]:
                best = (norm, b, cost)
        width = high - low
        low, high = best[1] - width / 100, best[1] + width / 100
    return best


def main() -> None:
    started = time.perf_counter()
    print(describe_environment())
    print(f"x0={X0} target={TARGET} n_steps={N_STEPS} activation=sigmoid")
    print(f"figure 6 window: w in {W_VALUES}, b in [{B_LOW}, {B_HIGH}]")
    print()

    print(f"--- crossing the wall at w={W_MAIN}, b step 1e-3")
    print(f"{'b':>10} {'cost':>13} {'||grad||':>15}")
    for i in range(21):
        b = -2.6200 + i * 1e-3
        cost, norm = gradient_norm_at(W_MAIN, b)
        print(f"{b:10.4f} {cost:13.8f} {norm:15.6e}")
    print()

    print(f"--- the flat, for contrast (w={W_MAIN})")
    flat_cost, flat_norm = gradient_norm_at(W_MAIN, FLAT_B)
    print(f"b={FLAT_B}: cost={flat_cost:.8f} ||grad||={flat_norm:.6e}")
    print()

    print("--- steepest point per w (nested grid, 4 rounds of 401 points)")
    print(f"{'w':>5} {'b at peak':>16} {'max ||grad||':>15} {'cost there':>13}")
    peaks = {}
    for w in W_VALUES:
        norm, b, cost = refine_peak(w, SEARCH_LOW, SEARCH_HIGH, 401, 4)
        peaks[w] = (norm, b, cost)
        print(f"{w:5.1f} {b:16.8f} {norm:15.6e} {cost:13.8f}")
    print()

    peak_norm = peaks[W_MAIN][0]
    print(f"ratio of steepest to flat at w={W_MAIN}: {peak_norm / flat_norm:.4e}")
    print(f"orders of magnitude: {math.log10(peak_norm / flat_norm):.4f}")
    print()
    print(f"elapsed {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
