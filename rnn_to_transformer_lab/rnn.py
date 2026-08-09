"""The plain recurrent network, in the form Pascanu et al. derive against.

The book keeps one notation for all sixteen chapters (appendix A). In it,
`x_t` is the input, `a_t` the pre-activation, `h_t = sigma(a_t)` the hidden
state, and the matrices are `W_xh`, `W_hh`, `W_hy`. Chapter 1 fixes that and
the chapter 1 code above already uses it.

Pascanu et al. write their recurrence as

    x_t = W_rec sigma(x_{t-1}) + W_in u_t + b

which looks like a different network and is not one. Their state variable is
the book's pre-activation, and the translation is exact rather than
approximate. Start from the familiar form,

    h_t = sigma(W_hh h_{t-1} + W_xh x_t + b_h),

name the argument of sigma as a_t, so that h_t = sigma(a_t), and substitute:

    a_t = W_hh sigma(a_{t-1}) + W_xh x_t + b_h.

That is their equation, symbol for symbol, with their x as the book's a and
their u as the book's x. Their footnote calls the two forms equivalent and
says theirs was chosen for convenience; the convenience is that the Jacobian
of the second is W_hh diag(sigma'(a_{t-1})), which factors the constant matrix
out of the state-dependent part. Every bound in the paper is built on that
factoring.

So this module carries the pre-activation as the state, and calls it `a`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

Activation = Callable[[torch.Tensor], torch.Tensor]


def tanh_prime(x: torch.Tensor) -> torch.Tensor:
    """Derivative of tanh, evaluated at x. Bounded by 1."""
    return 1.0 - torch.tanh(x) ** 2


def sigmoid_prime(x: torch.Tensor) -> torch.Tensor:
    """Derivative of the logistic sigmoid, evaluated at x. Bounded by 1/4."""
    s = torch.sigmoid(x)
    return s * (1.0 - s)


def identity(x: torch.Tensor) -> torch.Tensor:
    return x


def identity_prime(x: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(x)


#: The bound gamma on |sigma'(x)| for each activation the book uses. The
#: vanishing condition is stated against this number, so it is tabulated
#: rather than recomputed: for tanh gamma = 1, for the sigmoid gamma = 1/4.
GAMMA = {"tanh": 1.0, "sigmoid": 0.25, "identity": 1.0}

_ACTIVATIONS: dict[str, tuple[Activation, Activation]] = {
    "tanh": (torch.tanh, tanh_prime),
    "sigmoid": (torch.sigmoid, sigmoid_prime),
    "identity": (identity, identity_prime),
}


def activation(name: str) -> tuple[Activation, Activation]:
    """Look up (sigma, sigma') by name."""
    if name not in _ACTIVATIONS:
        raise KeyError(f"unknown activation {name!r}; have {sorted(_ACTIVATIONS)}")
    return _ACTIVATIONS[name]


@dataclass
class PlainRNN:
    """a_t = W_hh sigma(a_{t-1}) + W_xh x_t + b_h.

    Kept as a dataclass of plain tensors rather than an nn.Module on purpose.
    Chapter 3 is about what the Jacobians do, and every experiment here wants
    to reach in and set the spectrum of W_hh by hand. An nn.Module would add a
    parameter registry that nothing in this chapter uses.
    """

    w_hh: torch.Tensor
    w_xh: torch.Tensor
    b_h: torch.Tensor
    act: str = "tanh"

    @property
    def n_hidden(self) -> int:
        return self.w_hh.shape[0]

    def step(self, a_prev: torch.Tensor, x: torch.Tensor | None = None) -> torch.Tensor:
        """One step of the recurrence, from pre-activation to pre-activation."""
        sigma, _ = activation(self.act)
        out = self.w_hh @ sigma(a_prev) + self.b_h
        if x is not None:
            out = out + self.w_xh @ x
        return out

    def hidden(self, a: torch.Tensor) -> torch.Tensor:
        """h_t = sigma(a_t), the state in the book's notation."""
        sigma, _ = activation(self.act)
        return sigma(a)

    def unroll(
        self, a0: torch.Tensor, n_steps: int, inputs: torch.Tensor | None = None
    ) -> list[torch.Tensor]:
        """Run the recurrence and keep every pre-activation.

        Returns [a_0, a_1, ..., a_n_steps], so the list has n_steps + 1 entries
        and index t is the state at time t. The Jacobian helpers index it that
        way, and an off-by-one here is an off-by-one in every bound.
        """
        states = [a0]
        for t in range(n_steps):
            x = None if inputs is None else inputs[t]
            states.append(self.step(states[-1], x))
        return states

    def jacobian_at(self, a_prev: torch.Tensor) -> torch.Tensor:
        """d a_t / d a_{t-1} = W_hh diag(sigma'(a_{t-1})).

        Column-vector convention: entry (i, j) is d a_t[i] / d a_{t-1}[j].

        The paper's equation (5) prints this factor as
        W_rec^T diag(sigma'(x_{i-1})), because it propagates the gradient as a
        row vector back through time. Transposing this matrix properly gives
        diag(sigma'(a_{t-1})) W_hh^T, so the two expressions are not literally
        transposes of each other. Nothing downstream breaks: every bound in the
        paper is on the spectral norm, and a matrix and its transpose have the
        same one. Chapter 3 states the convention rather than inheriting the
        ambiguity.
        """
        _, sigma_prime = activation(self.act)
        return self.w_hh @ torch.diag(sigma_prime(a_prev))


def with_spectral_radius(w: torch.Tensor, radius: float) -> torch.Tensor:
    """Rescale w so that its spectral radius is exactly `radius`.

    The spectral radius is the largest |eigenvalue|. It is what the paper names
    in the statement of the vanishing condition, and it is not the same number
    as the largest singular value unless the matrix is normal. Chapter 3 turns
    on that difference, so the two are computed by separate functions here and
    never conflated.
    """
    eigenvalues = torch.linalg.eigvals(w)
    current = torch.max(torch.abs(eigenvalues)).real.item()
    if current == 0.0:
        raise ValueError("cannot rescale a nilpotent matrix to a given radius")
    return w * (radius / current)


def spectral_radius(w: torch.Tensor) -> float:
    """Largest |eigenvalue| of w."""
    return torch.max(torch.abs(torch.linalg.eigvals(w))).real.item()


def spectral_norm(w: torch.Tensor) -> float:
    """Largest singular value of w, which is the matrix 2-norm."""
    return torch.linalg.matrix_norm(w, ord=2).item()


def random_normal_matrix(n: int, radius: float, generator: torch.Generator) -> torch.Tensor:
    """A normal matrix with a given spectral radius.

    Normal means W W^T = W^T W, and for such a matrix the spectral radius and
    the spectral norm coincide. Built here as a random orthogonal matrix scaled
    by `radius`: orthogonal matrices are normal, all their singular values are
    1, and all their eigenvalues sit on the unit circle, so scaling moves both
    numbers together. This is the well-behaved case the paper's clean
    exponential describes.
    """
    a = torch.randn(n, n, generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    # The QR of a Gaussian matrix is only Haar-distributed after fixing the
    # sign convention; without this the diagonal of r biases the result.
    q = q * torch.sign(torch.diagonal(r))
    return q * radius


def jordan_block(n: int, eigenvalue: float, off_diagonal: float) -> torch.Tensor:
    """An upper triangular matrix: `eigenvalue` on the diagonal, `off_diagonal`
    once above it.

    Every eigenvalue equals `eigenvalue`, so the spectral radius is
    |eigenvalue| however large `off_diagonal` grows. The spectral norm grows
    with it. This is the cheapest matrix that separates the two numbers, and
    chapter 3 uses it to show that the paper's condition, read literally
    through eigenvalues, does not do what its own proof does.
    """
    w = torch.eye(n, dtype=torch.float64) * eigenvalue
    idx = torch.arange(n - 1)
    w[idx, idx + 1] = off_diagonal
    return w
