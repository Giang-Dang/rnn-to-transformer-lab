"""Chapter 4: what the derivative through the cell does, against distance.

Chapter 3 ended on a question. The gradient's path through time is a product
of Jacobians, and training keeps that product from decaying by penalising the
factors for leaving 1. Can the factor be 1 by construction instead?

This is the answer, measured. Three curves against the same distance axis:

* the plain RNN of chapter 3, spectral radius 0.9, tanh: the product decays
  geometrically and is the curve chapter 3 measured;
* the 1997 memory cell under the paper's truncation: the product is the
  identity at every distance, exactly, not approximately;
* the same cell with the truncated paths put back: the product is no longer 1,
  and this is the number that says what the truncation is actually doing.

Run: python experiments/ch04_flow.py
"""

from __future__ import annotations

import time

import torch

from rnn_to_transformer_lab.determinism import describe_environment, seed_everything
from rnn_to_transformer_lab.jacobians import product_norms
from rnn_to_transformer_lab.lstm import random_lstm
from rnn_to_transformer_lab.rnn import PlainRNN, random_normal_matrix

N_HIDDEN = 64
N_INPUT = 1
N_STEPS = 100
RADIUS = 0.9
#: Same distances chapter 3's decay table reports, so the two sit side by side.
REPORT_AT = (1, 10, 50, 100)
#: Weight scales for the cell. 0.1 is the interval the paper's Experiments 3-6
#: initialize in; 1.0 is far outside it, and is here because the question
#: "does the full derivative stay near 1 only because the weights are small"
#: has to be answered with a number rather than an opinion.
SCALES = (0.1, 1.0)


def lstm_product_norms(model, states, full: bool) -> list[float]:
    """Spectral norm of d c_t / d c_0 for every t, truncated or not."""
    n = model.n_hidden
    dtype = model.w_hi.dtype
    product = torch.eye(n, dtype=dtype)
    norms = [torch.linalg.matrix_norm(product, ord=2).item()]
    for t in range(1, len(states)):
        if full:
            step = model.cec_jacobian_full(states[t - 1], states[t])
        else:
            step = model.cec_jacobian_truncated()
        product = step @ product
        norms.append(torch.linalg.matrix_norm(product, ord=2).item())
    return norms


def main() -> None:
    started = time.perf_counter()
    seed_everything(0)
    print(describe_environment())
    print(f"n_hidden={N_HIDDEN} n_steps={N_STEPS} radius={RADIUS} (plain RNN)")
    print()

    generator = torch.Generator().manual_seed(0)
    a0 = torch.randn(N_HIDDEN, generator=generator, dtype=torch.float64) * 0.1
    gen_w = torch.Generator().manual_seed(1)
    w_hh = random_normal_matrix(N_HIDDEN, RADIUS, gen_w)
    rnn = PlainRNN(
        w_hh=w_hh,
        w_xh=torch.zeros(N_HIDDEN, 1, dtype=torch.float64),
        b_h=torch.zeros(N_HIDDEN, dtype=torch.float64),
        act="tanh",
    )
    rnn_norms = product_norms(rnn, rnn.unroll(a0, N_STEPS), k=0)

    rows: list[tuple[str, list[float]]] = [("plain RNN 0.9", rnn_norms)]
    for scale in SCALES:
        gen = torch.Generator().manual_seed(2)
        model = random_lstm(N_HIDDEN, N_INPUT, gen, scale=scale)
        inputs = torch.randn(N_STEPS, N_INPUT, generator=gen, dtype=torch.float64)
        states = model.unroll(inputs)
        if scale == SCALES[0]:
            rows.append(
                ("CEC truncated", lstm_product_norms(model, states, full=False))
            )
        rows.append(
            (f"CEC full w={scale}", lstm_product_norms(model, states, full=True))
        )

    header = "model            " + " ".join(f"{'l=' + str(l):<11}" for l in REPORT_AT)
    print(header)
    for label, norms in rows:
        cells = " ".join(f"{norms[l]:<11.4e}" for l in REPORT_AT)
        print(f"{label:<17}{cells}")

    print()
    truncated = rows[1][1]
    exact = all(value == 1.0 for value in truncated)
    print(f"truncated norm is exactly 1.0 at every distance: {exact}")
    print(f"largest deviation from 1.0: {max(abs(v - 1.0) for v in truncated):.1e}")

    print()
    print(f"elapsed {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
