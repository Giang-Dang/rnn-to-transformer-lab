"""Chapter 3: one step taken at the wall, with the clip and without it.

Same surface as ch03_surface.py. This finds the steepest point with the same
function that script uses, then takes a single gradient-descent step from it
twice, changing one argument.

The paper's figure 6 draws this as solid arrows leaving the valley and a dashed
arrow staying in it. The dashed arrow is the claim; the distance moved in
parameter space is the measurement.

Run: python experiments/ch03_clipping.py
"""

from __future__ import annotations

import math
import time

import torch

from rnn_to_transformer_lab.clipping import clip_norm
from rnn_to_transformer_lab.determinism import describe_environment
from rnn_to_transformer_lab.surface import cost_and_gradient, steepest_point

PEAK_W = 5.0
#: The same window ch03_surface.py searches. The peak is computed here rather
#: than pasted in, because a pasted constant drifts from the script that
#: produced it and then two experiments quietly describe two different points.
SEARCH_LOW, SEARCH_HIGH = -3.0, -2.0

LEARNING_RATE = 0.1
THRESHOLDS = (None, 1.0, 0.1)


def main() -> None:
    started = time.perf_counter()
    print(describe_environment())

    _, peak_b, _ = steepest_point(PEAK_W, SEARCH_LOW, SEARCH_HIGH)
    print(f"start point: w={PEAK_W} b={peak_b:.10f}")
    print(f"learning rate {LEARNING_RATE}")
    print()

    cost, grad_w, grad_b = cost_and_gradient(PEAK_W, peak_b)
    gradient = torch.tensor([grad_w, grad_b], dtype=torch.float64)
    norm = torch.linalg.vector_norm(gradient).item()
    print(f"cost at start      {cost:.8f}")
    print(f"dE/dw              {grad_w:.6e}")
    print(f"dE/db              {grad_b:.6e}")
    print(f"||grad||           {norm:.6e}")
    print()

    print(f"{'threshold':>10} {'fired':>6} {'step length':>14} {'w after':>16} {'b after':>16} {'cost after':>14}")
    for threshold in THRESHOLDS:
        used, fired = (gradient, False) if threshold is None else clip_norm(gradient, threshold)
        step = used * LEARNING_RATE
        step_length = torch.linalg.vector_norm(step).item()
        w_after = PEAK_W - step[0].item()
        b_after = peak_b - step[1].item()
        cost_after, _, _ = cost_and_gradient(w_after, b_after)
        label = "none" if threshold is None else f"{threshold}"
        print(
            f"{label:>10} {str(fired):>6} {step_length:14.6e} "
            f"{w_after:16.8f} {b_after:16.8f} {cost_after:14.8f}"
        )

    print()
    ratio = norm / 1.0
    print(f"unclipped step is {ratio:.4e} times the step clipped at 1.0")
    print(f"log10 of that ratio: {math.log10(ratio):.4f}")
    print()
    print(f"elapsed {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
