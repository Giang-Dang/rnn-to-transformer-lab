"""Chapter 10: the counts checked against real modules, the equivariance
claims checked against torch's own convolution.

Same discipline as chapter 9's tests. A closed form for a parameter count is
worth nothing until something confirms it counts *these* parameters, so
`layer_counts` is compared against `sum(p.numel())` on an `nn.Conv2d` and an
`nn.Linear` rather than against a second copy of the same arithmetic. The
equivariance tests assert exact zeros, because these are statements about the
arithmetic rather than about any trained weights - a tolerance here would hide
exactly the defect the chapter is about.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from rnn_to_transformer_lab.conv import (
    conv_as_attention_residual,
    equivariance_error,
    invariance_rate,
    layer_counts,
    lecun89_counts,
    lenet5_counts,
    shift,
)

SHAPES = ((32, 32, 3, 32), (16, 16, 8, 16), (28, 28, 1, 6))


@pytest.mark.parametrize("h,w,in_c,out_c", SHAPES)
@pytest.mark.parametrize("kernel", (3, 5))
def test_conv_regime_matches_a_built_conv2d(h, w, in_c, out_c, kernel):
    """The `conv` regime is the parameter count of an actual nn.Conv2d."""
    _, _, conv = layer_counts(h, w, in_c, h, w, out_c, kernel)
    built = nn.Conv2d(in_c, out_c, kernel, padding=kernel // 2)
    assert conv.parameters == sum(p.numel() for p in built.parameters())


@pytest.mark.parametrize("h,w,in_c,out_c", SHAPES)
def test_dense_regime_matches_a_built_linear(h, w, in_c, out_c):
    """The `dense` regime is the parameter count of the equivalent nn.Linear."""
    dense, _, _ = layer_counts(h, w, in_c, h, w, out_c, 3)
    built = nn.Linear(h * w * in_c, h * w * out_c)
    assert dense.parameters == sum(p.numel() for p in built.parameters())


def test_sharing_divisor_is_the_number_of_positions():
    """Sharing divides the weight count by however many places the layer looks.

    Stated as a test rather than as a comment because it is the chapter's
    headline reading of the two constraints, and it should fail loudly if the
    counting convention ever changes.
    """
    _, local, conv = layer_counts(32, 32, 3, 32, 32, 32, 3)
    assert local.weights == conv.weights * 32 * 32


def test_lecun89_rebuilds_the_published_totals():
    """Neural Computation 1(4):541-551, section 3.3, exactly."""
    layers, total = lecun89_counts()
    assert total.connections == 64_660
    assert total.parameters == 9_760
    assert total.sharing_ratio == 6.625
    # Section 3.3: "Units do not share their biases (thresholds)."
    h1 = layers[0]
    assert h1.biases == 768
    assert h1.weights == 300


def test_lenet5_rebuilds_and_60000_is_exact():
    """Proc. IEEE 86(11):2278-2324, section II.B.

    The output layer carries 840 connections and no free parameters, and the
    connection total does not close without it.
    """
    layers, total = lenet5_counts()
    assert total.connections == 340_908
    assert total.parameters == 60_000

    out = layers[-1]
    assert out.connections == 840
    assert out.parameters == 0
    without = sum(c.connections for c in layers[:-1])
    assert total.connections - without == 840


@pytest.mark.parametrize("dh,dw", ((1, 0), (0, 1), (3, 5), (7, 11)))
def test_circular_convolution_is_exactly_equivariant(dh, dw):
    """Exactly zero, not approximately. No tolerance on purpose."""
    torch.manual_seed(0)
    x = torch.randn(4, 3, 16, 16)
    layer = nn.Conv2d(3, 8, 3, padding=1, padding_mode="circular")
    assert equivariance_error(layer, x, dh, dw) == 0.0


def test_zero_padding_breaks_equivariance_only_at_the_border():
    torch.manual_seed(0)
    x = torch.randn(4, 3, 16, 16)
    layer = nn.Conv2d(3, 8, 3, padding=1)
    assert equivariance_error(layer, x, 1, 0) > 0.0
    with torch.no_grad():
        d = (layer(shift(x, 1, 0)) - shift(layer(x), 1, 0)).abs()
    assert d[:, :, 2:-2, 2:-2].max().item() == 0.0


@pytest.mark.parametrize("depth", (1, 2, 3))
def test_subsampling_is_exact_at_multiples_of_the_total_stride(depth):
    torch.manual_seed(0)
    x = torch.randn(4, 3, 32, 32)
    layers: list[nn.Module] = []
    channels = 3
    for _ in range(depth):
        layers += [
            nn.Conv2d(channels, 8, 3, padding=1, padding_mode="circular"),
            nn.MaxPool2d(2),
        ]
        channels = 8
    stack = nn.Sequential(*layers)
    stride = 2**depth
    assert equivariance_error(stack, x, stride, 0, downsample=stride) == 0.0
    assert equivariance_error(stack, x, 2 * stride, 0, downsample=stride) == 0.0


def test_a_shift_off_the_stride_grid_has_no_output_shift_to_compare():
    """The refusal is the finding, so it is asserted rather than worked around.

    A first draft of `equivariance_error` had no `downsample` argument and
    silently compared against the wrong output shift, which made a stride-2
    layer look non-equivariant at every shift including the ones where it is
    bit-exact. This test pins the corrected contract.
    """
    torch.manual_seed(0)
    x = torch.randn(2, 3, 16, 16)
    stack = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1, padding_mode="circular"), nn.MaxPool2d(2)
    )
    with pytest.raises(ValueError, match="not a multiple"):
        equivariance_error(stack, x, 1, 0, downsample=2)


def test_a_network_that_subsamples_is_not_shift_invariant():
    """Equivariance is not invariance, and the second claim is the false one."""
    torch.manual_seed(0)
    net = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(16 * 8 * 8, 10),
    )
    x = torch.randn(256, 3, 16, 16)
    assert invariance_rate(net, x, 1, 0) > 0.0


@pytest.mark.parametrize("kernel", (3, 5))
def test_k_squared_heads_reproduce_a_convolution(kernel):
    """Cordonnier et al. 2020, theorem 1, at the saturated limit.

    Float32 accumulation order is the only gap, so the tolerance is relative
    to the output scale rather than absolute.
    """
    torch.manual_seed(0)
    x = torch.randn(4, 3, 16, 16)
    w = torch.randn(8, 3, kernel, kernel)
    residual = conv_as_attention_residual(x, w)
    padded = nn.functional.pad(x, (kernel // 2,) * 4, mode="circular")
    scale = nn.functional.conv2d(padded, w).abs().max().item()
    assert residual / scale < 1e-5
