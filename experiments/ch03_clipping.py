"""Chapter 3: one step taken at the wall, with the clip and without it.

Same surface as ch03_surface.py. This starts a single gradient-descent step
from the steepest point found there and runs it twice, changing one argument.

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
from rnn_to_transformer_lab.surface import cost_and_gradient

#: The steepest point ch03_surface.py reports at w = 5.0. Hard-coded rather
#: than re-searched so that this script measures the step and nothing else; if
#: the surface script ever reports a different point, this constant is wrong
#: and verify.py says so.
PEAK_W = 5.0
PEAK_B = -2.6123413579

LEARNING_RATE = 0.1
THRESHOLDS = (None, 1.0, 0.1)


def main() -> None:
    started = time.perf_counter()
    print(describe_environment())
    print(f"start point: w={PEAK_W} b={PEAK_B}")
    print(f"learning rate {LEARNING_RATE}")
    print()

    cost, grad_w, grad_b = cost_and_gradient(PEAK_W, PEAK_B)
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
        b_after = PEAK_B - step[1].item()
        cost_after, _, _ = cost_and_gradient(w_after, b_after)
        label = "none" if threshold is None else f"{threshold}"
        print(
            f"{label:>10} {str(fired):>6} {step_length:14.6e} "
            f"{w_after:16.8f} {b_after:16.8f} {cost_after:14.8f}"
        )

    print()
    unclipped_step = LEARNING_RATE * norm
    print(f"unclipped step is {unclipped_step / (LEARNING_RATE * 1.0):.4e} times the step clipped at 1.0")
    print(f"log10 of that ratio: {math.log10(unclipped_step / (LEARNING_RATE * 1.0)):.4f}")
    print()
    print(f"elapsed {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
