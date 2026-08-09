"""Chapter 02: The adding problem, the copy task, and gradient norm decay.

Two classic synthetic tasks that expose the vanishing gradient problem in
plain RNNs.  This module trains a tanh-RNN on both tasks while logging the
gradient norm ||dL/dW_hh|| at each backward step, so the book can print
the decay curve.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Shared RNN helpers (reuse the same forward/backward from ch01, adapted)
# ---------------------------------------------------------------------------

def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def _rnn_forward(
    X: np.ndarray,
    W_xh: np.ndarray,
    W_hh: np.ndarray,
    W_hy: np.ndarray,
    b_h: np.ndarray,
    b_y: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Plain RNN forward pass.  Returns h, y, a, and per-step hidden states."""
    seq_len = X.shape[0]
    hidden_dim = W_hh.shape[0]
    h_all: list[np.ndarray] = [np.zeros(hidden_dim, dtype=np.float64)]
    y_all: list[np.ndarray] = []
    a_all: list[np.ndarray] = []

    for t in range(seq_len):
        x_t = X[t]
        a_t = W_xh @ x_t + W_hh @ h_all[-1] + b_h
        h_t = _tanh(a_t)
        y_t = W_hy @ h_t + b_y
        a_all.append(a_t)
        h_all.append(h_t)
        y_all.append(y_t)

    return h_all, y_all, a_all


def _rnn_bptt(
    X: np.ndarray,
    h_all: list[np.ndarray],
    y_all: list[np.ndarray],
    a_all: list[np.ndarray],
    W_hh: np.ndarray,
    W_hy: np.ndarray,
    dy: list[np.ndarray],
) -> tuple[dict[str, np.ndarray], list[float]]:
    """BPTT.  dy[t] = dL/dy_t.  Returns gradients and per-step ||dL/dW_hh||."""
    seq_len = X.shape[0]
    hidden_dim = W_hh.shape[0]

    dW_xh = np.zeros((hidden_dim, X.shape[1]), dtype=np.float64)
    dW_hh = np.zeros((hidden_dim, hidden_dim), dtype=np.float64)
    dW_hy = np.zeros((y_all[0].shape[0], hidden_dim), dtype=np.float64)
    db_h = np.zeros(hidden_dim, dtype=np.float64)
    db_y = np.zeros(y_all[0].shape[0], dtype=np.float64)

    dh_next = np.zeros(hidden_dim, dtype=np.float64)
    dWhh_norms: list[float] = []

    for t in reversed(range(seq_len)):
        dW_hy += np.outer(dy[t], h_all[t + 1])
        db_y += dy[t]

        dh_t = W_hy.T @ dy[t] + dh_next
        da_t = dh_t * (1.0 - h_all[t + 1] ** 2)

        dW_xh += np.outer(da_t, X[t])
        dW_hh += np.outer(da_t, h_all[t])
        db_h += da_t

        dh_next = W_hh.T @ da_t

        dWhh_norms.append(float(np.linalg.norm(dW_hh)))

    grads = {"W_xh": dW_xh, "W_hh": dW_hh, "W_hy": dW_hy,
             "b_h": db_h, "b_y": db_y}
    return grads, dWhh_norms


# ---------------------------------------------------------------------------
# Adding problem
# ---------------------------------------------------------------------------

def generate_adding(
    T: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one adding-problem sample of length T.

    Input: T×2 matrix.  Column 0 is uniform noise in [-1, 1].
    Column 1 is zero except at two random positions where it is 1.0.

    Target: scalar = sum of column 0 at the two marked positions.
    """
    seq = rng.uniform(-1.0, 1.0, size=(T,))
    mask = np.zeros(T, dtype=np.float64)
    i, j = rng.choice(T, size=2, replace=False)
    mask[i] = 1.0
    mask[j] = 1.0
    X = np.column_stack([seq, mask])
    target = np.array([seq[i] + seq[j]], dtype=np.float64)
    return X, target


def train_adding(
    T: int,
    hidden_dim: int = 24,
    lr: float = 0.01,
    n_samples: int = 10000,
    seed: int = 42,
) -> tuple[float, list[float], float]:
    """Train a plain RNN on the adding problem and return results.

    Returns:
        final_loss: MSE on the last 1000 samples
        dWhh_norms: per-backward-step ||dL/dW_hh|| on a fresh sample
        loss_before: loss before training (for comparison)
    """
    rng = np.random.default_rng(seed)
    input_dim = 2
    output_dim = 1

    # Init weights
    W_xh = rng.normal(0, 0.1, (hidden_dim, input_dim))
    W_hh = rng.normal(0, 0.1, (hidden_dim, hidden_dim))
    W_hy = rng.normal(0, 0.1, (output_dim, hidden_dim))
    b_h = np.zeros(hidden_dim, dtype=np.float64)
    b_y = np.zeros(output_dim, dtype=np.float64)

    # Measure loss before training
    X_test, target_test = generate_adding(T, rng)
    _, y_test, _ = _rnn_forward(X_test, W_xh, W_hh, W_hy, b_h, b_y)
    loss_before = float(np.mean((y_test[-1] - target_test) ** 2))

    # Train
    for step in range(n_samples):
        X, target = generate_adding(T, rng)
        h_all, y_all, a_all = _rnn_forward(X, W_xh, W_hh, W_hy, b_h, b_y)

        # dy[t] = 0 for t < T-1; only the last output matters
        dy = [np.zeros(output_dim, dtype=np.float64) for _ in range(T)]
        dy[-1] = y_all[-1] - target

        grads, _ = _rnn_bptt(X, h_all, y_all, a_all, W_hh, W_hy, dy)

        for key in grads:
            grads[key] /= T  # average gradient

        W_xh -= lr * grads["W_xh"]
        W_hh -= lr * grads["W_hh"]
        W_hy -= lr * grads["W_hy"]
        b_h -= lr * grads["b_h"]
        b_y -= lr * grads["b_y"]

    # Final loss
    losses = []
    for _ in range(1000):
        X_t, tgt = generate_adding(T, rng)
        _, y_t, _ = _rnn_forward(X_t, W_xh, W_hh, W_hy, b_h, b_y)
        losses.append(float((y_t[-1] - tgt) ** 2))
    final_loss = float(np.mean(losses))

    # Gradient norm on a fresh sample
    X_fresh, tgt_fresh = generate_adding(T, rng)
    h_f, y_f, a_f = _rnn_forward(X_fresh, W_xh, W_hh, W_hy, b_h, b_y)
    dy_f = [np.zeros(output_dim) for _ in range(T)]
    dy_f[-1] = y_f[-1] - tgt_fresh
    _, dWhh_norms = _rnn_bptt(X_fresh, h_f, y_f, a_f, W_hh, W_hy, dy_f)

    return final_loss, dWhh_norms, loss_before


# ---------------------------------------------------------------------------
# Copy task
# ---------------------------------------------------------------------------

def generate_copy(
    T_mem: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one copy-task sample.

    Input:  T_mem values in {0,1}^8 (one-hot), then a blank of zeros with
    a 1.0 delimiter, then T_mem zeros.  Total length = 2*T_mem + 1.
    The output is the original T_mem values, asked only at the last T_mem
    timesteps (before that, loss is not computed / target is don't-care).

    Returns:
        X: (2*T_mem + 1) × 9  (8 data bits + 1 delimiter channel)
        targets: T_mem × 8
    """
    n_bits = 8
    total_len = 2 * T_mem + 1

    data = rng.integers(0, 2, size=(T_mem, n_bits)).astype(np.float64)
    X = np.zeros((total_len, n_bits + 1), dtype=np.float64)

    # First T_mem steps: data bits + delimiter=0
    X[:T_mem, :n_bits] = data
    # Step T_mem: delimiter = 1.0, zeros in data
    X[T_mem, n_bits] = 1.0
    # Last T_mem steps: zeros (deliberately left as zeros)

    return X, data


def train_copy(
    T_mem: int,
    hidden_dim: int = 32,
    lr: float = 0.01,
    n_epochs: int = 3000,
    seed: int = 42,
) -> tuple[float, list[float], float]:
    """Train a plain RNN on the copy task.

    Returns:
        final_loss: cross-entropy per bit on last 100 samples
        dWhh_norms: per-step ||dL/dW_hh|| on a fresh sample
        loss_before: loss before training
    """
    rng = np.random.default_rng(seed)
    input_dim = 9   # 8 data bits + 1 delimiter
    output_dim = 8
    total_len = 2 * T_mem + 1

    W_xh = rng.normal(0, 0.1, (hidden_dim, input_dim))
    W_hh = rng.normal(0, 0.1, (hidden_dim, hidden_dim))
    W_hy = rng.normal(0, 0.1, (output_dim, hidden_dim))
    b_h = np.zeros(hidden_dim, dtype=np.float64)
    b_y = np.zeros(output_dim, dtype=np.float64)

    def _softmax_ce(y_pred: np.ndarray, y_true: np.ndarray) -> tuple[float, np.ndarray]:
        """Stable softmax + cross-entropy. Returns loss and dy."""
        y_pred = y_pred - np.max(y_pred)
        exp_y = np.exp(y_pred)
        softmax_y = exp_y / np.sum(exp_y)
        eps = 1e-12
        loss = -np.sum(y_true * np.log(softmax_y + eps))
        dy = softmax_y - y_true
        return float(loss), dy

    # Measure loss before training
    X_test, tgt_test = generate_copy(T_mem, rng)
    _, y_test, _ = _rnn_forward(X_test, W_xh, W_hh, W_hy, b_h, b_y)
    loss_before = 0.0
    for i in range(T_mem):
        l, _ = _softmax_ce(y_test[T_mem + 1 + i], tgt_test[i])
        loss_before += l
    loss_before /= T_mem

    for epoch in range(n_epochs):
        X, targets = generate_copy(T_mem, rng)
        h_all, y_all, a_all = _rnn_forward(X, W_xh, W_hh, W_hy, b_h, b_y)

        # Only compute loss on the last T_mem outputs
        dy = [np.zeros(output_dim, dtype=np.float64) for _ in range(total_len)]
        total_loss = 0.0
        for i in range(T_mem):
            t = T_mem + 1 + i
            l, d = _softmax_ce(y_all[t], targets[i])
            total_loss += l
            dy[t] = d

        grads, _ = _rnn_bptt(X, h_all, y_all, a_all, W_hh, W_hy, dy)
        for key in grads:
            grads[key] /= T_mem

        W_xh -= lr * grads["W_xh"]
        W_hh -= lr * grads["W_hh"]
        W_hy -= lr * grads["W_hy"]
        b_h -= lr * grads["b_h"]
        b_y -= lr * grads["b_y"]

    # Final loss
    losses = []
    for _ in range(100):
        X_t, tgt = generate_copy(T_mem, rng)
        _, y_t, _ = _rnn_forward(X_t, W_xh, W_hh, W_hy, b_h, b_y)
        l = 0.0
        for i in range(T_mem):
            ll, _ = _softmax_ce(y_t[T_mem + 1 + i], tgt[i])
            l += ll
        losses.append(l / T_mem)
    final_loss = float(np.mean(losses))

    # Gradient norm on a fresh sample
    X_fresh, tgt_fresh = generate_copy(T_mem, rng)
    h_f, y_f, a_f = _rnn_forward(X_fresh, W_xh, W_hh, W_hy, b_h, b_y)
    dy_f = [np.zeros(output_dim) for _ in range(total_len)]
    for i in range(T_mem):
        t = T_mem + 1 + i
        _, d = _softmax_ce(y_f[t], tgt_fresh[i])
        dy_f[t] = d
    _, dWhh_norms = _rnn_bptt(X_fresh, h_f, y_f, a_f, W_hh, W_hy, dy_f)

    return final_loss, dWhh_norms, loss_before


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify() -> None:
    """Run the full verification for chapter 02."""
    print("=== Chapter 02: Adding problem & copy task ===\n")

    # Adding problem at various T
    print("--- Adding problem ---")
    for T in [10, 20, 50]:
        loss, norms, loss0 = train_adding(T, n_samples=5000)
        norm_last = norms[0] if norms else 0  # norm at step T (closest to output)
        norm_first = norms[-1] if len(norms) > 1 else norms[0]  # norm at step 1
        ratio = norm_first / norm_last if norm_last > 1e-12 else float('inf')
        print(f"  T={T:3d}: loss={loss:.4f} (before={loss0:.4f}), "
              f"||dW_hh|| at t=1: {norm_first:.6f}, "
              f"at t=T: {norm_last:.6f}, "
              f"ratio 1/T: {ratio:.2f}")

    # Copy task at various T_mem
    print("\n--- Copy task ---")
    for T_mem in [5, 10, 20]:
        loss, norms, loss0 = train_copy(T_mem, n_epochs=2000)
        norm_last = norms[0] if norms else 0
        norm_first = norms[-1] if len(norms) > 1 else norms[0]
        ratio = norm_first / norm_last if norm_last > 1e-12 else float('inf')
        print(f"  T_mem={T_mem:2d}: loss={loss:.4f} (before={loss0:.4f}), "
              f"||dW_hh|| at first: {norm_first:.6f}, "
              f"at last: {norm_last:.6f}, "
              f"ratio first/last: {ratio:.2f}")

    # Sanity checks
    # 1. Training should reduce loss vs before-training
    loss10, _, loss0_10 = train_adding(10, n_samples=5000)
    assert loss10 < loss0_10 * 0.8, f"Adding T=10 should improve: {loss0_10} -> {loss10}"

    # 2. Larger T should be harder (higher loss or less improvement)
    loss50, _, loss0_50 = train_adding(50, n_samples=5000)
    improvement_10 = loss0_10 / max(loss10, 1e-8)
    improvement_50 = loss0_50 / max(loss50, 1e-8)
    assert improvement_10 > improvement_50, \
        f"T=10 should improve more than T=50: {improvement_10:.1f}x vs {improvement_50:.1f}x"

    print("\nAll chapter 02 checks passed.")


if __name__ == "__main__":
    verify()
