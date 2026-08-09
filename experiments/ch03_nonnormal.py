"""Chapter 3: the gap between the largest eigenvalue and the largest singular value.

The paper states its sufficient condition for vanishing in terms of lambda_1,
"the absolute value of the largest eigenvalue" of W_rec. Its proof, equation
(6), bounds ||W_rec^T|| ||diag(sigma'(x_k))||, and the first of those is a
singular value. For a normal matrix the two coincide and the difference is
invisible. For a matrix that is not normal they are different numbers, and only
the singular-value reading makes the proof go through.

This runs the linear case, sigma set to the identity, which is the setting of
the paper's own supplementary proof. There the Jacobian product is exactly
W^l, equation (12), so nothing here depends on a trajectory or on saturation.

Run: python experiments/ch03_nonnormal.py
"""

from __future__ import annotations

import time

from rnn_to_transformer_lab.determinism import describe_environment, seed_everything
from rnn_to_transformer_lab.jacobians import linear_power_norms, transient_growth
from rnn_to_transformer_lab.rnn import jordan_block, spectral_norm, spectral_radius

MAX_POWER = 100
EIGENVALUE = 0.9
OFF_DIAGONAL = 3.0
REPORT_AT = (1, 5, 9, 10, 20, 50, 100)


def main() -> None:
    started = time.perf_counter()
    seed_everything(0)
    print(describe_environment())
    print()

    w = jordan_block(2, EIGENVALUE, OFF_DIAGONAL)
    print(f"W = [[{EIGENVALUE}, {OFF_DIAGONAL}], [0, {EIGENVALUE}]]")
    print(f"spectral radius (largest |eigenvalue|) = {spectral_radius(w):.6f}")
    print(f"spectral norm   (largest singular value) = {spectral_norm(w):.6f}")
    print()

    norms = linear_power_norms(w, MAX_POWER)
    peak_at, peak_value = transient_growth(norms)
    print("l     ||W^l||")
    for l in REPORT_AT:
        print(f"{l:<5} {norms[l]:.6e}")
    print()
    print(f"peak at l={peak_at}, ||W^l||={peak_value:.6f}")
    print(f"amplification over l=1: {peak_value / norms[1]:.4f}")
    print(f"first l with ||W^l|| < 1: {next(l for l, v in enumerate(norms) if v < 1.0 and l > 0)}")
    print()
    print(f"elapsed {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
