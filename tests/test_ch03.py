"""Every number chapter 3 prints, asserted.

The rule this file exists to enforce: if the book prints a decimal, something
here fails when that decimal stops being true. Tolerances are tight on purpose.
These are deterministic computations on a fixed seed, not a training run, so a
value that moves by more than the tolerance means something changed that the
chapter should know about.

If a later chapter legitimately changes one of these, change the assertion and
say so in the commit. Never widen a tolerance to make a run pass.
"""

from __future__ import annotations

import math

import pytest
import torch

from rnn_to_transformer_lab.clipping import clip_norm
from rnn_to_transformer_lab.determinism import seed_everything
from rnn_to_transformer_lab.jacobians import decay_rate, linear_power_norms, product_norms, transient_growth
from rnn_to_transformer_lab.regularizer import omega, step_ratios
from rnn_to_transformer_lab.rnn import (
    GAMMA,
    PlainRNN,
    jordan_block,
    random_normal_matrix,
    spectral_norm,
    spectral_radius,
)
from rnn_to_transformer_lab.surface import cost_and_gradient

REL = 1e-4


def build(radius: float, n_hidden: int, seed: int = 1) -> PlainRNN:
    gen = torch.Generator().manual_seed(seed)
    return PlainRNN(
        w_rec=random_normal_matrix(n_hidden, radius, gen),
        w_in=torch.zeros(n_hidden, 1, dtype=torch.float64),
        bias=torch.zeros(n_hidden, dtype=torch.float64),
        act="tanh",
    )


def initial_state(n_hidden: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(0)
    return torch.randn(n_hidden, generator=gen, dtype=torch.float64) * 0.1


# --- the bound itself -------------------------------------------------------


def test_gamma_values_match_the_paper():
    """tanh has gamma = 1, the sigmoid gamma = 1/4."""
    assert GAMMA["tanh"] == 1.0
    assert GAMMA["sigmoid"] == 0.25


def test_orthogonal_matrix_is_normal_so_radius_equals_norm():
    """The case where the paper's eigenvalue wording and its proof agree."""
    for radius in (0.5, 0.9, 1.0, 1.2):
        w = random_normal_matrix(64, radius, torch.Generator().manual_seed(1))
        assert spectral_radius(w) == pytest.approx(radius, rel=REL)
        assert spectral_norm(w) == pytest.approx(radius, rel=REL)


def test_jacobian_factors_as_w_rec_times_diag_sigma_prime():
    """Equation (5), checked against autograd rather than against itself."""
    seed_everything(0)
    model = build(0.9, 8)
    x_prev = initial_state(8)
    analytic = model.jacobian_at(x_prev)
    numeric = torch.autograd.functional.jacobian(
        lambda x: model.step(x), x_prev.clone().requires_grad_(True)
    )
    assert torch.allclose(analytic, numeric, atol=1e-10)


# --- ch03_decay.py ----------------------------------------------------------


@pytest.mark.parametrize(
    "radius, norm_at_100, rate",
    [
        (0.5, 7.8786e-31, 0.5000),
        (0.9, 2.5902e-05, 0.9000),
        (1.0, 6.2852e-01, 0.9961),
        (1.2, 8.2505e01, 1.0244),
    ],
)
def test_decay_against_spectral_radius(radius, norm_at_100, rate):
    seed_everything(0)
    model = build(radius, 64)
    states = model.unroll(initial_state(64), 100)
    norms = product_norms(model, states, k=0)
    assert norms[100] == pytest.approx(norm_at_100, rel=1e-3)
    assert decay_rate(norms, 50, 100) == pytest.approx(rate, abs=5e-5)


def test_saturation_holds_the_exploding_case_far_below_its_radius():
    """A radius of 1.2 does not give 1.2 per step.

    The measured rate is a little over 1, because tanh'(x) falls away from 1 as
    soon as the state leaves the origin. This is why the paper's exploding
    condition is only necessary: passing it does not make gradients explode.
    """
    seed_everything(0)
    model = build(1.2, 64)
    states = model.unroll(initial_state(64), 100)
    measured = decay_rate(product_norms(model, states, k=0), 50, 100)
    assert measured > 1.0
    assert measured < 1.05
    assert measured == pytest.approx(1.0244, abs=5e-5)


# --- ch03_nonnormal.py ------------------------------------------------------


def test_jordan_block_separates_radius_from_norm():
    w = jordan_block(2, 0.9, 3.0)
    assert spectral_radius(w) == pytest.approx(0.9, rel=REL)
    assert spectral_norm(w) == pytest.approx(3.249286, rel=REL)


def test_gradient_grows_for_nine_steps_under_a_radius_below_one():
    """The claim chapter 3 rests its correction on.

    Spectral radius 0.9, so the paper's stated sufficient condition for
    vanishing is satisfied. The Jacobian product still grows by a factor of
    3.58 before it turns, and stays above 1 until step 48.
    """
    w = jordan_block(2, 0.9, 3.0)
    norms = linear_power_norms(w, 100)
    peak_at, peak_value = transient_growth(norms)
    assert peak_at == 9
    assert peak_value == pytest.approx(11.635514, rel=REL)
    assert peak_value / norms[1] == pytest.approx(3.5809, rel=1e-3)
    first_below_one = next(l for l, v in enumerate(norms) if v < 1.0 and l > 0)
    assert first_below_one == 49
    assert norms[100] == pytest.approx(8.853879e-03, rel=REL)


# --- ch03_surface.py --------------------------------------------------------


@pytest.mark.parametrize(
    "b, cost, grad_norm",
    [
        (-2.6130, 0.34157104, 2.942246e-01),
        (-2.6120, 0.00847633, 9.281034e00),
        (-2.0000, 0.05558549, 5.549566e-02),
    ],
)
def test_surface_points(b, cost, grad_norm):
    measured_cost, grad_w, grad_b = cost_and_gradient(5.0, b)
    assert measured_cost == pytest.approx(cost, rel=REL)
    assert math.hypot(grad_w, grad_b) == pytest.approx(grad_norm, rel=1e-3)


def test_the_wall_is_one_grid_step_wide():
    """The cost falls by a third over a step of 0.001 in b."""
    left, _, _ = cost_and_gradient(5.0, -2.6130)
    right, _, _ = cost_and_gradient(5.0, -2.6120)
    assert left - right == pytest.approx(0.33309471, rel=1e-3)


def test_steepest_point_and_its_distance_from_the_flat():
    """Five orders of magnitude between the top of the wall and the flat."""
    _, gw, gb = cost_and_gradient(5.0, -2.6123413579)
    peak = math.hypot(gw, gb)
    _, fw, fb = cost_and_gradient(5.0, -2.0)
    flat = math.hypot(fw, fb)
    assert peak == pytest.approx(6.773125e03, rel=1e-3)
    assert flat == pytest.approx(5.549566e-02, rel=1e-3)
    assert math.log10(peak / flat) == pytest.approx(5.0865, abs=1e-3)


# --- ch03_clipping.py -------------------------------------------------------


def test_clip_norm_rescales_without_turning():
    gradient = torch.tensor([-3.0, -4.0], dtype=torch.float64)
    clipped, fired = clip_norm(gradient, 1.0)
    assert fired
    assert torch.linalg.vector_norm(clipped).item() == pytest.approx(1.0, rel=1e-12)
    cosine = torch.dot(clipped, gradient) / (
        torch.linalg.vector_norm(clipped) * torch.linalg.vector_norm(gradient)
    )
    assert cosine.item() == pytest.approx(1.0, rel=1e-12)


def test_clip_norm_is_a_no_op_below_the_threshold():
    gradient = torch.tensor([0.3, 0.4], dtype=torch.float64)
    clipped, fired = clip_norm(gradient, 1.0)
    assert not fired
    assert torch.equal(clipped, gradient)


def test_the_unclipped_step_leaves_the_surface():
    """One step at the wall, without the clip, and where it lands."""
    _, grad_w, grad_b = cost_and_gradient(5.0, -2.6123413579)
    w_after = 5.0 - 0.1 * grad_w
    b_after = -2.6123413579 - 0.1 * grad_b
    assert w_after == pytest.approx(362.77215206, rel=1e-3)
    assert b_after == pytest.approx(572.49753715, rel=1e-3)
    cost_after, _, _ = cost_and_gradient(w_after, b_after)
    # The unit saturates, so sigmoid(x_50) is 1 and the cost is exactly
    # (1 - 0.7)^2. The step did not find a worse minimum; it left the problem.
    assert cost_after == pytest.approx(0.09, abs=1e-12)


@pytest.mark.parametrize(
    "threshold, w_after, b_after, cost_after",
    [
        (1.0, 5.05282231, -2.52743080, 0.02586525),
        (0.1, 5.00528223, -2.60385030, 0.01215804),
    ],
)
def test_the_clipped_step_stays_and_improves(threshold, w_after, b_after, cost_after):
    cost_before, grad_w, grad_b = cost_and_gradient(5.0, -2.6123413579)
    gradient = torch.tensor([grad_w, grad_b], dtype=torch.float64)
    clipped, fired = clip_norm(gradient, threshold)
    assert fired
    step = clipped * 0.1
    new_w = 5.0 - step[0].item()
    new_b = -2.6123413579 - step[1].item()
    assert new_w == pytest.approx(w_after, rel=1e-6)
    assert new_b == pytest.approx(b_after, rel=1e-6)
    measured, _, _ = cost_and_gradient(new_w, new_b)
    assert measured == pytest.approx(cost_after, rel=1e-5)
    assert measured < cost_before


# --- ch03_regularizer.py ----------------------------------------------------


@pytest.mark.parametrize(
    "radius, mean_ratio, total",
    [
        (0.5, 0.499463, 7.516259),
        (0.9, 0.895816, 0.326728),
        (1.0, 0.978175, 0.014876),
    ],
)
def test_regularizer_terms(radius, mean_ratio, total):
    seed_everything(0)
    model = build(radius, 32)
    x0 = initial_state(32).requires_grad_(True)
    states = model.unroll(x0, 30)
    cost = 0.5 * torch.sum(states[-1] ** 2)
    grads = torch.autograd.grad(cost, states, allow_unused=True)
    signals = [torch.zeros_like(states[i]) if g is None else g for i, g in enumerate(grads)]
    ratios = step_ratios(model, states, signals)
    assert sum(ratios) / len(ratios) == pytest.approx(mean_ratio, rel=1e-4)
    assert omega(model, states, signals).item() == pytest.approx(total, rel=1e-4)


def test_omega_falls_as_the_recurrence_gets_closer_to_norm_preserving():
    """The penalty is monotone in the thing it is meant to measure."""
    totals = []
    for radius in (0.5, 0.9, 1.0):
        seed_everything(0)
        model = build(radius, 32)
        x0 = initial_state(32).requires_grad_(True)
        states = model.unroll(x0, 30)
        cost = 0.5 * torch.sum(states[-1] ** 2)
        grads = torch.autograd.grad(cost, states, allow_unused=True)
        signals = [torch.zeros_like(states[i]) if g is None else g for i, g in enumerate(grads)]
        totals.append(omega(model, states, signals).item())
    assert totals[0] > totals[1] > totals[2]
