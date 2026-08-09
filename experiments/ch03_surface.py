"""Chapter 3: the wall in the error surface, measured rather than drawn.

Reproduces the setting of figure 6 in Pascanu et al.: one hidden unit, no
input, a_t = w sigmoid(a_{t-1}) + b, a_0 = 0.5, and the cost read once at step
50 as (sigmoid(a_50) - 0.7)^2.

The figure shows a wall. What a figure cannot show is how narrow it is, and
that is the number the chapter needs: the gradient norm at the top of the wall
against the gradient norm a few tenths away on the flat.

Run: python experiments/ch03_surface.py
"""

from __future__ import annotations

import math
import time

from rnn_to_transformer_lab.determinism import describe_environment
from rnn_to_transformer_lab.surface import (
    N_STEPS,
    TARGET,
    X0,
    coarse_scan_max,
    cost_and_gradient,
    steepest_point,
)

#: The window figure 6 plots.
W_VALUES = (4.6, 4.8, 5.0, 5.2, 5.4)
B_LOW, B_HIGH = -2.8, -2.0
#: Widened for the peak search only: at w = 5.4 the wall has moved just past
#: the left edge of the window the paper drew.
SEARCH_LOW, SEARCH_HIGH = -3.0, -2.0
FLAT_B = -2.0
W_MAIN = 5.0
COARSE_STEP = 0.01


def gradient_norm_at(w: float, b: float) -> tuple[float, float]:
    cost, grad_w, grad_b = cost_and_gradient(w, b)
    return cost, math.hypot(grad_w, grad_b)


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

    print(f"--- what a step of {COARSE_STEP} finds instead, over the whole window")
    coarse_norm, coarse_b = coarse_scan_max(W_MAIN, B_LOW, B_HIGH, COARSE_STEP)
    print(f"max ||grad||={coarse_norm:.6e} at b={coarse_b:.4f}")
    print("the grid steps over the wall; this is a different number, not a rough one")
    print()

    print("--- steepest point per w (nested grid, 4 rounds of 401 points)")
    print(f"{'w':>5} {'b at peak':>18} {'max ||grad||':>15} {'cost there':>13}")
    peaks = {}
    for w in W_VALUES:
        norm, b, cost = steepest_point(w, SEARCH_LOW, SEARCH_HIGH)
        peaks[w] = (norm, b, cost)
        print(f"{w:5.1f} {b:18.10f} {norm:15.6e} {cost:13.8f}")
    print()

    print("--- the same search kept inside the window the paper drew")
    inside_norm, inside_b, _ = steepest_point(5.4, B_LOW, B_HIGH)
    print(f"w=5.4 over b in [{B_LOW}, {B_HIGH}]: max ||grad||={inside_norm:.6e} at b={inside_b:.8f}")
    print("the wall at w=5.4 sits outside that window, so this number means nothing")
    print()

    peak_norm = peaks[W_MAIN][0]
    print(f"ratio of steepest to flat at w={W_MAIN}: {peak_norm / flat_norm:.4e}")
    print(f"orders of magnitude: {math.log10(peak_norm / flat_norm):.4f}")
    print()
    print(f"elapsed {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
