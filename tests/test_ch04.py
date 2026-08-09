"""Every number chapter 4 prints, asserted.

Same rule as chapter 3: if the book prints a decimal, something here fails when
that decimal stops being true. Tolerances are tight because these are
deterministic computations on fixed seeds.

Two assertions here are exact rather than approximate, and they are meant to
be. The truncated carousel Jacobian is the identity matrix, not a matrix close
to it, and a cell whose input gate is shut carries its state bit for bit rather
than nearly. Both are architectural facts, and a tolerance on either would hide
a bug rather than absorb float noise.

What is deliberately *not* asserted exactly: c_t - c_{t-1} against i_t g(a_c).
That one is true of the architecture and false of floating point, because
(a + b) - a is not b. The exact form of the same claim is the shut-gate test.
"""

from __future__ import annotations

import torch

from rnn_to_transformer_lab.determinism import seed_everything
from rnn_to_transformer_lab.lstm import (
    Lstm1997,
    LstmForget,
    fully_connected_parameters,
    g_in,
    g_out,
    gru_parameters,
    layer_parameters,
    random_lstm,
)

REL = 1e-4


def build(n_hidden: int = 5, n_input: int = 3, scale: float = 0.8, seed: int = 1):
    gen = torch.Generator().manual_seed(seed)
    model = random_lstm(n_hidden, n_input, gen, scale=scale)
    inputs = torch.randn(9, n_input, generator=gen, dtype=torch.float64)
    return model, inputs


# --- the squashing functions the paper actually specifies -------------------


def test_squashing_ranges_match_appendix_a1():
    """g has range [-2, 2] and g_out has range [-1, 1]. Neither is tanh."""
    wide = torch.linspace(-40.0, 40.0, 20001, dtype=torch.float64)
    assert g_in(wide).min().item() >= -2.0
    assert g_in(wide).max().item() <= 2.0
    assert g_out(wide).min().item() >= -1.0
    assert g_out(wide).max().item() <= 1.0
    # The bounds are approached, not merely respected: at |x| = 40 the logistic
    # is within 1e-17 of its limit and float64 rounds the result onto it.
    assert g_in(torch.tensor(-40.0, dtype=torch.float64)).item() == -2.0
    assert g_out(torch.tensor(40.0, dtype=torch.float64)).item() == 1.0
    # Inside the range the functions are strictly between the bounds.
    narrow = torch.linspace(-6.0, 6.0, 101, dtype=torch.float64)
    assert g_in(narrow).abs().max().item() < 2.0
    # g is not tanh. Reaching for tanh out of habit is the likeliest way to
    # reimplement this cell wrongly, and the gap is not small: over [-6, 6]
    # the two differ by more than 0.8 at their furthest apart.
    assert (g_in(narrow) - torch.tanh(narrow)).abs().max().item() > 0.8


def test_squashing_functions_are_scaled_logistics():
    x = torch.linspace(-6.0, 6.0, 101, dtype=torch.float64)
    assert torch.allclose(g_in(x), 2.0 * g_out(x))
    assert torch.allclose(g_out(x), torch.tanh(x / 2.0))


# --- the 1997 cell has no forget gate ---------------------------------------


def test_the_1997_update_has_no_forget_gate():
    """c_t = c_{t-1} + i_t g(a_c): the self-connection is a fixed 1.0.

    The increment is compared with a tolerance rather than bit-exactly,
    because (a + b) - a is not b in floating point. The coefficient on
    c_{t-1} is checked exactly instead, in the test below, where it can be.
    """
    model, inputs = build()
    states = model.unroll(inputs)
    for t in range(1, len(states)):
        increment = states[t].c - states[t - 1].c
        assert torch.allclose(
            increment, states[t].i * g_in(states[t].a_c), rtol=0, atol=1e-14
        )


def test_a_closed_input_gate_carries_the_state_bit_for_bit():
    """The exact form of the claim: with i_t = 0 the state does not move.

    A forget gate would multiply c_{t-1} by something below 1 here and the
    state would decay. The 1997 cell holds it, to the last bit, which is what
    "constant" in constant error carousel is claiming.
    """
    gen = torch.Generator().manual_seed(3)
    model = random_lstm(6, 2, gen, scale=0.5)
    # A large negative input-gate bias closes the gate; -800 puts sigmoid at
    # exactly 0.0 in float64 rather than merely near it.
    model.b_i = torch.full((6,), -800.0, dtype=torch.float64)
    inputs = torch.randn(50, 2, generator=gen, dtype=torch.float64)
    states = model.unroll(inputs)
    assert states[1].i.max().item() == 0.0
    for t in range(2, len(states)):
        assert torch.equal(states[t].c, states[t - 1].c)


def test_the_forget_gate_variant_does_have_one():
    gen = torch.Generator().manual_seed(1)
    base = random_lstm(4, 2, gen, scale=0.5)
    model = LstmForget(
        **{f.name: getattr(base, f.name) for f in base.__dataclass_fields__.values()},
        w_xf=torch.zeros(4, 2, dtype=torch.float64),
        w_hf=torch.zeros(4, 4, dtype=torch.float64),
        b_f=torch.zeros(4, dtype=torch.float64),
    )
    inputs = torch.randn(6, 2, generator=gen, dtype=torch.float64)
    states = model.unroll(inputs)
    # Zero forget-gate weights and bias put f_t at sigmoid(0) = 0.5, so the
    # previous state is halved every step rather than carried.
    for t in range(1, len(states)):
        assert torch.allclose(states[t].f, torch.full((4,), 0.5, dtype=torch.float64))
        expected = 0.5 * states[t - 1].c + states[t].i * g_in(states[t].a_c)
        assert torch.allclose(states[t].c, expected)


# --- the carousel -----------------------------------------------------------


def test_truncated_carousel_jacobian_is_exactly_the_identity():
    model, _ = build()
    jacobian = model.cec_jacobian_truncated()
    assert torch.equal(jacobian, torch.eye(5, dtype=torch.float64))


def test_truncated_product_norm_is_one_at_every_distance():
    """The whole point of the architecture, over 100 steps rather than one."""
    model, _ = build(n_hidden=8, n_input=2)
    gen = torch.Generator().manual_seed(4)
    inputs = torch.randn(100, 2, generator=gen, dtype=torch.float64)
    states = model.unroll(inputs)
    product = torch.eye(8, dtype=torch.float64)
    for _ in range(1, len(states)):
        product = model.cec_jacobian_truncated() @ product
        assert torch.linalg.matrix_norm(product, ord=2).item() == 1.0


def test_analytic_full_jacobian_matches_autograd():
    """The one derivation in chapter 4 a reader is most likely to redo."""
    model, inputs = build()
    states = model.unroll(inputs)
    t = 6
    previous = states[t - 1]

    def c_of_c_prev(c_prev: torch.Tensor) -> torch.Tensor:
        h_prev = previous.o * g_out(c_prev)
        return model.step(c_prev, h_prev, inputs[t - 1]).c

    auto = torch.autograd.functional.jacobian(c_of_c_prev, previous.c.clone())
    analytic = model.cec_jacobian_full(previous, states[t])
    assert (auto - analytic).abs().max().item() < 1e-12


def test_full_jacobian_is_not_the_identity():
    """Otherwise the truncation would be measuring nothing."""
    model, inputs = build()
    states = model.unroll(inputs)
    full = model.cec_jacobian_full(states[5], states[6])
    assert (full - torch.eye(5, dtype=torch.float64)).abs().max().item() > 1e-3


# --- truncation -------------------------------------------------------------


def test_truncation_leaves_the_forward_pass_alone():
    model, inputs = build()
    plain = model.unroll(inputs, truncate=False)
    cut = model.unroll(inputs, truncate=True)
    for a, b in zip(plain, cut):
        assert torch.equal(a.c, b.c)
        assert torch.equal(a.h, b.h)


def test_truncation_still_reaches_the_recurrent_weights():
    """It cuts the path back through time, not the weight update itself.

    If truncation zeroed the gradient on W_hi the algorithm would not be
    learning those connections at all, which is a different algorithm from the
    paper's.
    """
    gen = torch.Generator().manual_seed(2)
    model = random_lstm(6, 2, gen, scale=0.3, requires_grad=True)
    inputs = torch.randn(20, 2, generator=gen, dtype=torch.float64)
    states = model.unroll(inputs, truncate=True)
    loss = states[-1].h.sum()
    grads = torch.autograd.grad(loss, [model.w_hi, model.w_hc, model.w_ho])
    for g in grads:
        assert g.abs().max().item() > 0.0


# --- the input weight conflict ----------------------------------------------


def test_ungated_carousel_optimum_is_one_over_T():
    """The closed form behind chapter 4's conflict table.

    c_T = w sum_t x_t against target x_1 is one least-squares fit in one
    variable, and its solution is 1/T. Monte Carlo, so the tolerance is loose;
    the point is the trend, not the fourth decimal.
    """
    seed_everything(7)
    gen = torch.Generator().manual_seed(7)
    for lag in (10, 50, 100):
        inputs = torch.randn(20000, lag, generator=gen, dtype=torch.float64)
        target = inputs[:, 0]
        totals = inputs.sum(dim=1)
        w_star = (totals @ target / (totals @ totals)).item()
        assert abs(w_star - 1.0 / lag) < 0.25 / lag
        residual = totals * w_star - target
        mse = (residual @ residual / 20000).item()
        variance = (target @ target / 20000).item()
        # Explains a fraction 1/T of its own target's variance, and no more.
        assert abs((1.0 - mse / variance) - 1.0 / lag) < 0.25 / lag


# --- parameter counts -------------------------------------------------------


def test_block_counts_are_three_and_four():
    plain = layer_parameters(256, 256, blocks=1)
    assert layer_parameters(256, 256, blocks=3) == 3 * plain
    assert layer_parameters(256, 256, blocks=4) == 4 * plain
    assert gru_parameters(256, 256) == 3 * plain


def test_the_papers_own_topology_gives_nine_on_the_recurrent_block():
    one = fully_connected_parameters(64, 8)
    three = fully_connected_parameters(192, 8)
    assert (192**2) / (64**2) == 9.0
    # The whole layer lands below 9 because input weights and biases scale by
    # 3 rather than by 9. The chapter prints this number.
    assert abs(three / one - 8.2603) < 1e-3


def test_repo_cell_and_torch_lstm_agree_on_the_block_count():
    gen = torch.Generator().manual_seed(0)
    mine = sum(p.numel() for p in random_lstm(32, 16, gen).parameters())
    assert mine == layer_parameters(32, 16, blocks=3)
    reference = torch.nn.LSTM(16, 32, num_layers=1, bias=True)
    theirs = sum(p.numel() for p in reference.parameters())
    # torch carries two bias vectors per block rather than one.
    assert theirs == layer_parameters(32, 16, blocks=4) + 4 * 32


# --- the numbers the chapter prints -----------------------------------------


def test_flow_table_reproduces_chapter_three_for_the_plain_rnn():
    """Chapter 4's comparison row has to be chapter 3's measurement.

    If these drift apart, one of the two chapters is printing a number the
    other one contradicts, which is worse than either being wrong alone.
    """
    from rnn_to_transformer_lab.jacobians import product_norms
    from rnn_to_transformer_lab.rnn import PlainRNN, random_normal_matrix

    gen_w = torch.Generator().manual_seed(1)
    w_hh = random_normal_matrix(64, 0.9, gen_w)
    rnn = PlainRNN(
        w_hh=w_hh,
        w_xh=torch.zeros(64, 1, dtype=torch.float64),
        b_h=torch.zeros(64, dtype=torch.float64),
        act="tanh",
    )
    gen = torch.Generator().manual_seed(0)
    a0 = torch.randn(64, generator=gen, dtype=torch.float64) * 0.1
    norms = product_norms(rnn, rnn.unroll(a0, 100), k=0)
    for distance, printed in ((1, 9.0000e-01), (10, 3.4200e-01),
                              (50, 5.0259e-03), (100, 2.5902e-05)):
        assert abs(norms[distance] - printed) < REL * printed


def test_logistic_derivative_bound_forces_a_self_weight_of_four():
    """Chapter 4 prints 0.25 and 4.0; both come from this."""
    from rnn_to_transformer_lab.rnn import sigmoid_prime

    best = sigmoid_prime(torch.linspace(-8, 8, 100001, dtype=torch.float64)).max().item()
    assert abs(best - 0.25) < 1e-9
    assert abs(1.0 / best - 4.0) < 1e-7
