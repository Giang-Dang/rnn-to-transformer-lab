"""Chapter 01: Plain RNN, BPTT with Jacobians, finite-difference gradient check.

This module implements a vanilla RNN from scratch using only numpy, derives
BPTT by hand in Jacobian form, and verifies the result against finite
differences.  No `nn.RNN`, no autograd — the point is to see every
intermediate Jacobian.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Fixed toy problem: 3 timesteps, input dim 2, hidden dim 3, output dim 2
# ---------------------------------------------------------------------------
# Weights and inputs are small fixed arrays so every number in the book is
# reproducible.  They were drawn from a fixed seed but are stored literally
# so the listing is self-contained and does not depend on numpy's rng state.

INPUT_DIM = 2
HIDDEN_DIM = 3
OUTPUT_DIM = 2
SEQ_LEN = 3

# Input sequence: shape (SEQ_LEN, INPUT_DIM) = (3, 2)
X: np.ndarray = np.array(
    [[0.50, -0.30],
     [0.80, 0.10],
     [-0.40, 0.60]],
    dtype=np.float64,
)

# Targets: shape (SEQ_LEN, OUTPUT_DIM) = (3, 2)
TARGET: np.ndarray = np.array(
    [[1.0, 0.0],
     [0.5, 0.5],
     [0.0, 1.0]],
    dtype=np.float64,
)

# Weights initialised from a fixed seed (seed=42, numpy 2.2)
W_xh: np.ndarray = np.array(
    [[0.374540, 0.950714],
     [0.731994, 0.598658],
     [0.156019, 0.155995]],
    dtype=np.float64,
)

W_hh: np.ndarray = np.array(
    [[0.058084, 0.866176, 0.601115],
     [0.708073, 0.020584, 0.969910],
     [0.832443, 0.212339, 0.181825]],
    dtype=np.float64,
)

W_hy: np.ndarray = np.array(
    [[0.183405, 0.304242, 0.524756],
     [0.431945, 0.291229, 0.611853]],
    dtype=np.float64,
)

b_h: np.ndarray = np.array([0.139494, 0.292145, 0.366362], dtype=np.float64)
b_y: np.ndarray = np.array([0.456070, 0.785176], dtype=np.float64)


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def forward() -> tuple[
    list[np.ndarray],  # h: hidden states (including h0)
    list[np.ndarray],  # y: outputs
    list[np.ndarray],  # a: pre-activations
    float,             # loss
]:
    """Run the RNN forward pass over the 3-step sequence.

    Returns:
        h: list of hidden states [h0, h1, h2, h3], each shape (HIDDEN_DIM,)
        y: list of outputs [y1, y2, y3], each shape (OUTPUT_DIM,)
        a: list of pre-activations [a1, a2, a3], each shape (HIDDEN_DIM,)
        loss: scalar 0.5 * sum_t ||y_t - target_t||^2
    """
    h = [np.zeros(HIDDEN_DIM, dtype=np.float64)]  # h0 = 0
    y: list[np.ndarray] = []
    a: list[np.ndarray] = []

    for t in range(SEQ_LEN):
        x_t = X[t]
        h_prev = h[-1]
        a_t = W_xh @ x_t + W_hh @ h_prev + b_h
        h_t = tanh(a_t)
        y_t = W_hy @ h_t + b_y
        a.append(a_t)
        h.append(h_t)
        y.append(y_t)

    loss = 0.0
    for t in range(SEQ_LEN):
        diff = y[t] - TARGET[t]
        loss += 0.5 * np.sum(diff * diff)

    return h, y, a, loss


def backward(
    h: list[np.ndarray],
    y: list[np.ndarray],
    a: list[np.ndarray],
) -> dict[str, np.ndarray]:
    """BPTT in Jacobian form.

    Args:
        h: hidden states [h0, h1, h2, h3]
        y: outputs [y1, y2, y3]
        a: pre-activations [a1, a2, a3]

    Returns:
        Gradients: dW_xh, dW_hh, dW_hy, db_h, db_y
    """
    # Accumulators
    dW_xh = np.zeros_like(W_xh)
    dW_hh = np.zeros_like(W_hh)
    dW_hy = np.zeros_like(W_hy)
    db_h = np.zeros_like(b_h)
    db_y = np.zeros_like(b_y)

    # dh_next carries the gradient signal from t+1 back to t
    dh_next = np.zeros(HIDDEN_DIM, dtype=np.float64)

    # Walk backwards through time
    for t in reversed(range(SEQ_LEN)):
        # Gradient from the output at this timestep
        dy_t = y[t] - TARGET[t]  # ∂L/∂y_t

        # dW_hy and db_y accumulate at every step
        dW_hy += np.outer(dy_t, h[t + 1])  # ∂L/∂y_t ⊗ h_t^T
        db_y += dy_t

        # Total gradient w.r.t. h_t: from output + from future
        dh_t = W_hy.T @ dy_t + dh_next  # (2)

        # Through tanh:  da_t = dh_t ⊙ (1 - h_t^2)
        da_t = dh_t * (1.0 - h[t + 1] * h[t + 1])  # (3)

        # Accumulate parameter gradients
        dW_xh += np.outer(da_t, X[t])               # ∂L/∂a_t ⊗ x_t^T
        dW_hh += np.outer(da_t, h[t])               # ∂L/∂a_t ⊗ h_{t-1}^T
        db_h += da_t

        # Pass gradient to previous timestep through W_hh
        dh_next = W_hh.T @ da_t                      # (4)

    return {
        "W_xh": dW_xh,
        "W_hh": dW_hh,
        "W_hy": dW_hy,
        "b_h": db_h,
        "b_y": db_y,
    }


def finite_difference(
    eps: float = 1e-5,
) -> dict[str, np.ndarray]:
    """Compute gradients via two-sided finite differences.

    Perturbs each scalar parameter independently and measures the change in
    loss.  Returns the same dict shape as backward().
    """
    # Collect all parameter arrays and their names
    params: list[tuple[str, np.ndarray]] = [
        ("W_xh", W_xh),
        ("W_hh", W_hh),
        ("W_hy", W_hy),
        ("b_h", b_h),
        ("b_y", b_y),
    ]

    grads: dict[str, np.ndarray] = {}

    for name, P in params:
        G = np.zeros_like(P)
        it = np.nditer(P, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index

            # Perturb +
            old = P[idx]
            P[idx] = old + eps
            _, _, _, loss_plus = forward()
            P[idx] = old - eps
            _, _, _, loss_minus = forward()
            P[idx] = old

            G[idx] = (loss_plus - loss_minus) / (2.0 * eps)

        grads[name] = G

    return grads


def verify() -> None:
    """Run the full verification for chapter 01.

    Prints the forward pass results, the manual gradients, the finite
    difference gradients, and the relative error between the two.
    Exits with code 1 if any check fails.
    """
    h, y, a, loss = forward()

    print("=== Forward pass ===")
    for t in range(SEQ_LEN):
        print(f"  x_{t+1} = {X[t]}")
        print(f"  h_{t+1} = {np.round(h[t+1], 6)}")
        print(f"  y_{t+1} = {np.round(y[t], 6)}")
        print(f"  target_{t+1} = {TARGET[t]}")
    print(f"  loss = {loss:.10f}")

    manual = backward(h, y, a)
    fd = finite_difference()

    print("\n=== Gradient check ===")
    all_ok = True
    for name in ["W_xh", "W_hh", "W_hy", "b_h", "b_y"]:
        m = manual[name]
        f = fd[name]
        abs_diff = np.max(np.abs(m - f))
        denom = np.max(np.abs(m)) + np.max(np.abs(f))
        rel_err = abs_diff / denom if denom > 0 else abs_diff
        status = "OK" if rel_err < 1e-4 else "FAIL"
        if rel_err >= 1e-4:
            all_ok = False
        print(f"  {name}: max |manual - fd| = {abs_diff:.2e}, "
              f"relative = {rel_err:.2e}  [{status}]")

    # Sanity checks on forward pass
    assert h[0].shape == (HIDDEN_DIM,) and np.all(h[0] == 0.0), "h0 must be zero"
    assert len(h) == SEQ_LEN + 1, "wrong number of hidden states"
    assert len(y) == SEQ_LEN, "wrong number of outputs"
    assert loss > 0.0, "loss should be positive"

    if not all_ok:
        raise SystemExit(1)

    print("\nAll gradient checks passed.")


if __name__ == "__main__":
    verify()
