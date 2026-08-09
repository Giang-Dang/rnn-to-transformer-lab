"""Chapter 4: the input weight conflict, with a closed form to check against.

The paper's section 3.2 names the reason a bare carousel is not enough. One
input weight has to do two jobs: switch the unit on when the input matters, and
leave it alone when the input does not. Both jobs pull on the same number, and
because the unit is linear the two pulls do not resolve into a useful
compromise.

That is measurable exactly, without training anything. Take a naive cell,

    c_t = c_{t-1} + w x_t,   read out c_T,   target x_1,

with x_t drawn independently, mean zero, variance v. Then c_T = w sum_t x_t and
the whole problem is one least-squares fit in one variable:

    w* = E[S x_1] / E[S^2] = v / (T v) = 1 / T,
    MSE(w*) = v (1 - 1/T).

So the best the naive cell can do is explain a fraction 1/T of the variance of
its own target, and the fraction goes to zero as the lag it is supposed to
bridge grows. Not a slow optimum: the best one.

An input gate changes the arithmetic rather than the optimizer's luck. A gate
that is open at step 1 and closed afterwards gives c_T = w x_1, and then w = 1
is exact. The gate here is set by hand, not learned; the claim being checked is
that the architecture can express the solution, not that it finds it.

Run: python experiments/ch04_conflict.py
"""

from __future__ import annotations

import time

import torch

from rnn_to_transformer_lab.determinism import describe_environment, seed_everything

LAGS = (10, 50, 100)
N_SEQUENCES = 20000
SEED = 7


def naive_cell(inputs: torch.Tensor, w: float) -> torch.Tensor:
    """c_T for a carousel with one ungated input weight: w times the sum."""
    return w * inputs.sum(dim=1)


def gated_cell(inputs: torch.Tensor, w: float) -> torch.Tensor:
    """c_T when an input gate is open at step 1 and closed after it."""
    return w * inputs[:, 0]


def main() -> None:
    started = time.perf_counter()
    seed_everything(SEED)
    print(describe_environment())
    print(f"n_sequences={N_SEQUENCES} inputs ~ N(0, 1), target = x_1")
    print()

    generator = torch.Generator().manual_seed(SEED)

    print("    T  w* measured  w* = 1/T    MSE at w*   var(target)  explained")
    for lag in LAGS:
        inputs = torch.randn(N_SEQUENCES, lag, generator=generator, dtype=torch.float64)
        target = inputs[:, 0]
        totals = inputs.sum(dim=1)
        # Least squares in one variable, solved rather than searched.
        w_star = (totals @ target / (totals @ totals)).item()
        residual = naive_cell(inputs, w_star) - target
        mse = (residual @ residual / N_SEQUENCES).item()
        variance = (target @ target / N_SEQUENCES).item()
        print(
            f"{lag:5d}  {w_star:<11.6f} {1.0 / lag:<11.6f} "
            f"{mse:<11.6f} {variance:<12.6f} {1.0 - mse / variance:.6f}"
        )

    print()
    print("the same cell with an input gate open only at step 1")
    print("    T  MSE at w=1")
    for lag in LAGS:
        inputs = torch.randn(N_SEQUENCES, lag, generator=generator, dtype=torch.float64)
        target = inputs[:, 0]
        residual = gated_cell(inputs, 1.0) - target
        print(f"{lag:5d}  {(residual @ residual / N_SEQUENCES).item():.6e}")

    print()
    print(f"elapsed {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
