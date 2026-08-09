"""Chapter 4: the naive constant error flow, and why the weight has to be 1.

Section 3.2 of the paper asks what it takes for error to flow unchanged through
a single unit with a single self-connection. The answer is one equation,

    f'(net_j(t)) w_jj = 1.0 for every t,

and integrating it forces f to be linear and the self-weight to be 1.0. Two
measurements here, because the equation is easy to read as a preference rather
than as the constraint it is.

First: a linear unit, self-weight near 1. The error flow over q steps is w^q,
so the cost of being slightly wrong is exponential in the distance the error
has to travel. This is the argument for fixing the weight rather than learning
it.

Second: a logistic unit. Its derivative never exceeds 1/4, so w >= 4 is
necessary before the product can reach 1 at all, and at w = 4 the equality
holds at exactly one state. Everywhere else the unit is back to decaying.

Run: python experiments/ch04_cec.py
"""

from __future__ import annotations

import time

import torch

from rnn_to_transformer_lab.determinism import describe_environment, seed_everything
from rnn_to_transformer_lab.rnn import sigmoid_prime

N_STEPS = 100
REPORT_AT = (1, 10, 50, 100)
LINEAR_WEIGHTS = (0.9, 0.99, 1.0, 1.01, 1.1)
#: 1/4 is the largest value the logistic derivative takes, so 4 is the smallest
#: self-weight for which f'(net) w = 1 is reachable at all. 1 and 2 are below
#: it, 8 is past it.
LOGISTIC_WEIGHTS = (1.0, 2.0, 4.0, 8.0)


def linear_flow(w: float) -> list[float]:
    """Error flow over q steps through a linear unit with self-weight w."""
    flow = [1.0]
    for _ in range(N_STEPS):
        flow.append(flow[-1] * w)
    return flow


def logistic_flow(w: float, y0: float = 0.5) -> tuple[list[float], list[float]]:
    """Per-step factor f'(net_t) w, and the running product, for a logistic unit.

    net_t = w y_{t-1}, y_t = sigmoid(net_t), with no external input and no
    bias, so the unit runs to its own fixed point and the factor is measured
    along the trajectory the unit actually takes.
    """
    y = torch.tensor(y0, dtype=torch.float64)
    factors: list[float] = []
    flow = [1.0]
    for _ in range(N_STEPS):
        net = w * y
        factor = (sigmoid_prime(net) * w).item()
        factors.append(factor)
        flow.append(flow[-1] * factor)
        y = torch.sigmoid(net)
    return factors, flow


def main() -> None:
    started = time.perf_counter()
    seed_everything(0)
    print(describe_environment())
    print(f"n_steps={N_STEPS}")
    print()

    print("linear unit, error flow w^q")
    print("     w  " + " ".join(f"{'q=' + str(q):<11}" for q in REPORT_AT))
    for w in LINEAR_WEIGHTS:
        cells = " ".join(f"{linear_flow(w)[q]:<11.4e}" for q in REPORT_AT)
        print(f"{w:6.2f}  {cells}")

    print()
    print("logistic unit, per-step factor f'(net) w and its product")
    print("     w  factor t=1   factor t=100 product q=100")
    for w in LOGISTIC_WEIGHTS:
        factors, flow = logistic_flow(w)
        print(f"{w:6.2f}  {factors[0]:<12.6f} {factors[-1]:<12.6f} {flow[-1]:.4e}")

    print()
    best = max(sigmoid_prime(torch.linspace(-8, 8, 100001, dtype=torch.float64))).item()
    print(f"largest value of the logistic derivative: {best:.8f}")
    print(f"smallest self-weight that can reach 1.0:  {1.0 / best:.6f}")

    print()
    print(f"elapsed {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
