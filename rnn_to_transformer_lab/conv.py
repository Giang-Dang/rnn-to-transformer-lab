"""The inductive bias of a convolutional layer, counted and measured.

Chapter 10. Two questions this module answers, and they are different in kind.

**What a convolution costs in parameters**, which is arithmetic. A convolution
is a fully connected layer with two constraints laid on it: each output unit
may look at a small neighborhood (locality), and units that look at different
places must use the same weights (sharing). Neither constraint is a new kind of
layer, and `layer_counts` prints the same layer under all three regimes so the
two constraints can be priced separately. `lecun89_counts` and `lenet5_counts`
then rebuild the published totals of the two papers that introduced the idea,
in the same spirit as chapter 9's rebuild of BERT and GPT - except that these
two come back exactly, which is the interesting part.

**What a convolution buys in equivariance**, which is a property of the
arithmetic rather than of any trained model, so it can be measured at
initialization and it is bit-exact. `shift`, `equivariance_error` and
`invariance_rate` are the three things chapter 10 measures. The result worth
knowing in advance: convolution with circular padding is exactly
shift-equivariant, zero padding is exactly equivariant except at the border,
and a stride-2 layer is equivariant to even shifts and not to odd ones. That
last one is the whole of Zhang (2019), and LeCun et al. (1989) already named
it - "the input image is undersampled and some position information is
eliminated" - about their own two-pixel subsampling.

**Equivariance is not invariance and this module keeps them apart.** A layer is
shift-equivariant when shifting the input shifts the output the same way; a
network is shift-invariant when shifting the input leaves the output alone. The
first is a theorem about convolution, the second is a claim about a whole
network, and the second is false for the networks people build. Vaswani-era
prose routinely says "translation invariance" where the property being used is
equivariance - so does a good deal of the vision literature - and the
distinction is the point of the chapter's second section.

Nothing here is timed. The counting half is closed form and identical on every
machine; the equivariance half runs on fixed seeds at initialization and is
reproducible to the digit for a pinned torch build.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# The two constraints, counted
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LayerCount:
    """One layer's size, under one of the three regimes.

    `connections` is the number of edges in the graph, which is what the layer
    costs to evaluate. `weights` and `biases` are the free parameters, which is
    what it costs to store and to learn. Weight sharing changes the second
    without changing the first, and that gap is the whole idea.
    """

    regime: str
    connections: int
    weights: int
    biases: int

    @property
    def parameters(self) -> int:
        return self.weights + self.biases

    @property
    def sharing_ratio(self) -> float:
        """Connections per free parameter. 1.0 when nothing is shared.

        Infinite for a layer with no free parameters at all, which is not a
        degenerate case invented to be tidy: LeNet-5's output layer is exactly
        that, 840 connections whose weights the paper fixes by hand and never
        trains.
        """
        if self.parameters == 0:
            return float("inf")
        return self.connections / self.parameters


def layer_counts(
    in_h: int,
    in_w: int,
    in_c: int,
    out_h: int,
    out_w: int,
    out_c: int,
    kernel: int,
    *,
    share_biases: bool = True,
) -> tuple[LayerCount, LayerCount, LayerCount]:
    """The same layer under all three regimes: dense, local, local and shared.

    Returned in that order, so a table reads as two constraints applied one
    after the other. `share_biases` is False for the LeCun 1989 topology, which
    shares weights across a feature map but gives every unit its own bias; that
    choice is not cosmetic and section 3.3 of the paper states it outright.

    Connections are identical for `local` and `conv`: sharing removes free
    parameters and removes no edges.
    """
    out_units = out_h * out_w * out_c
    fan_in_dense = in_h * in_w * in_c
    fan_in_local = kernel * kernel * in_c

    dense = LayerCount(
        regime="dense",
        connections=out_units * (fan_in_dense + 1),
        weights=out_units * fan_in_dense,
        biases=out_units,
    )
    local = LayerCount(
        regime="local",
        connections=out_units * (fan_in_local + 1),
        weights=out_units * fan_in_local,
        biases=out_units,
    )
    conv = LayerCount(
        regime="conv",
        connections=out_units * (fan_in_local + 1),
        weights=out_c * fan_in_local,
        biases=out_c if share_biases else out_units,
    )
    return dense, local, conv


def lecun89_counts() -> tuple[list[LayerCount], LayerCount]:
    """Rebuild LeCun et al. (1989), layer by layer, from the paper's own text.

    Neural Computation 1(4):541-551, section 3.3. Every quantity below is
    stated in the paper, and the paper writes its own arithmetic out - "only
    1068 free parameters (768 biases plus 25 times 12 feature kernels)" - so
    this is a check that the stated totals follow from the stated layers, not a
    reconstruction of anything the paper left out.

    Input is 16x16. H1 is 12 feature maps of 8x8, each unit reading a 5x5
    neighborhood two pixels apart (the subsampling is in the stride, not in a
    pooling layer). H2 is 12 maps of 4x4, each unit reading 5x5 from each of 8
    of H1's 12 maps. H3 is 30 units fully connected to H2, and the output is 10
    units fully connected to H3.

    Returns the four layers and their total. The total the paper prints is
    1256 units, 64,660 connections and 9,760 independent parameters.
    """
    # H1: 768 units, 25 input lines plus a bias each, 12 shared 5x5 kernels.
    h1 = LayerCount("H1", connections=768 * 26, weights=25 * 12, biases=768)
    # H2: 192 units, 200 input lines plus a bias, 12 maps of 200 shared weights.
    h2 = LayerCount("H2", connections=192 * 201, weights=200 * 12, biases=192)
    # H3: 30 units fully connected to H2's 192. Nothing is shared from here on.
    h3 = LayerCount("H3", connections=30 * 192 + 30, weights=30 * 192, biases=30)
    out = LayerCount("output", connections=10 * 30 + 10, weights=10 * 30, biases=10)

    layers = [h1, h2, h3, out]
    total = LayerCount(
        "total",
        connections=sum(x.connections for x in layers),
        weights=sum(x.weights for x in layers),
        biases=sum(x.biases for x in layers),
    )
    return layers, total


def lenet5_counts() -> tuple[list[LayerCount], LayerCount]:
    """Rebuild LeNet-5's trainable parameters from LeCun et al. (1998).

    Proc. IEEE 86(11):2278-2324, section II.B. The paper prints 340,908
    connections and "only 60,000 trainable free parameters", and both come back
    exactly - the second is not a round number that happens to be close, it is
    the sum.

    C3's connection scheme is the awkward one: it is not fully connected to S2.
    Table I of the paper gives the map, and it comes to 1516 parameters, which
    is the number used here.

    **The output layer is why a naive sum misses.** LeNet-5 ends in 10 RBF
    units of 84 inputs each, which is 840 connections, and the paper's total
    counts them. Their weight vectors are fixed by hand rather than learned, so
    they contribute nothing to the 60,000. Leave the layer out and the
    parameters still come to 60,000 while the connections come to 340,068,
    exactly 840 short - which is how this row was found.
    """
    c1 = LayerCount("C1", connections=122_304, weights=6 * 25, biases=6)
    s2 = LayerCount("S2", connections=5_880, weights=6, biases=6)
    c3 = LayerCount("C3", connections=151_600, weights=1_516 - 16, biases=16)
    s4 = LayerCount("S4", connections=2_000, weights=16, biases=16)
    c5 = LayerCount("C5", connections=48_120, weights=48_120 - 120, biases=120)
    f6 = LayerCount("F6", connections=10_164, weights=84 * 120, biases=84)
    out = LayerCount("output", connections=84 * 10, weights=0, biases=0)

    layers = [c1, s2, c3, s4, c5, f6, out]
    total = LayerCount(
        "total",
        connections=sum(x.connections for x in layers),
        weights=sum(x.weights for x in layers),
        biases=sum(x.biases for x in layers),
    )
    return layers, total


# ---------------------------------------------------------------------------
# Equivariance, measured
# ---------------------------------------------------------------------------


def shift(x: torch.Tensor, dh: int, dw: int) -> torch.Tensor:
    """Shift an NCHW batch cyclically by (dh, dw).

    Cyclic on purpose. A shift that pads with zeros is two operations - a shift
    and an erasure - and mixing them makes the border effect look like a
    failure of equivariance when it is a failure of the shift. The chapter
    measures the border effect separately, by changing the layer's padding
    rather than the shift.
    """
    return torch.roll(x, shifts=(dh, dw), dims=(2, 3))


def equivariance_error(
    layer: nn.Module, x: torch.Tensor, dh: int, dw: int, *, downsample: int = 1
) -> float:
    """max |layer(shift(x, d)) - shift(layer(x), d/downsample)| over the batch.

    Zero means the layer commutes with that shift exactly. This is a property
    of the layer's arithmetic, so it holds at initialization and needs no
    training; a nonzero value at float32 is either a genuine asymmetry (a
    border, a stride) or accumulation noise, and the chapter's tables report
    the number rather than a verdict so the two can be told apart by size.

    **`downsample` is not a convenience and getting it wrong inverts the
    result.** A layer that halves the resolution answers an input shift of 2
    with an output shift of 1, so comparing against an output shift of 2 makes
    the layer look non-equivariant at *every* shift, even the ones where it is
    exact. A first draft of this function had no such parameter and reported
    exactly that, which reads as a much stronger claim than the true one and is
    wrong. An input shift that is not a multiple of `downsample` has no
    corresponding output shift at all; that is the real defect, and it is what
    Zhang (2019) is about.
    """
    if dh % downsample or dw % downsample:
        raise ValueError(
            f"shift ({dh}, {dw}) is not a multiple of downsample={downsample}; "
            "there is no output shift to compare against, which is the finding "
            "rather than an error to work around - measure it with "
            "`invariance_rate` instead"
        )
    with torch.no_grad():
        lhs = layer(shift(x, dh, dw))
        rhs = shift(layer(x), dh // downsample, dw // downsample)
        return (lhs - rhs).abs().max().item()


def invariance_rate(model: nn.Module, x: torch.Tensor, dh: int, dw: int) -> float:
    """Fraction of inputs whose predicted class changes under one shift.

    The whole-network counterpart of `equivariance_error`, and the quantity
    that is actually claimed when someone says a convolutional network is
    "translation invariant". A rate above zero refutes the claim for that
    network at that shift.
    """
    with torch.no_grad():
        before = model(x).argmax(dim=1)
        after = model(shift(x, dh, dw)).argmax(dim=1)
        return (before != after).float().mean().item()


def conv_from_attention_weights(
    x: torch.Tensor, offsets: list[tuple[int, int]]
) -> torch.Tensor:
    """Gather one fixed neighbor per head, the way Cordonnier et al. construct.

    Their theorem 1 says a multi-head self-attention layer with N_h heads can
    express any convolution of kernel size sqrt(N_h) by sqrt(N_h), and the
    construction is that each head's attention distribution collapses onto one
    fixed offset. This function is the limit of that construction with the
    softmax already saturated: it returns, for each head, the input shifted by
    that head's offset. Stacking the results and mixing them with a per-head
    value matrix is then literally a convolution, which is what
    `experiments/ch10_equivariance.py` checks against `F.conv2d`.

    It is deliberately not a self-attention layer. Building one whose softmax
    actually saturates needs the paper's quadratic positional encoding driven
    to a large enough coefficient, and the residual then reports how far the
    softmax got rather than whether the theorem is true. The chapter says on
    the page that this is the construction's limit and not a trained model.
    """
    return torch.stack([shift(x, -dh, -dw) for dh, dw in offsets], dim=0)


def conv_as_attention_residual(
    x: torch.Tensor, weight: torch.Tensor
) -> float:
    """max |conv2d(x, weight) - the head-gather form of the same convolution|.

    `weight` is (out_c, in_c, K, K). Circular padding on the convolution side,
    because the gather side is cyclic; with both cyclic the two are the same
    arithmetic and the residual should be at float32 rounding rather than at
    any structural gap.
    """
    out_c, in_c, k, _ = weight.shape
    radius = k // 2
    offsets = [
        (i - radius, j - radius) for i in range(k) for j in range(k)
    ]
    gathered = conv_from_attention_weights(x, offsets)

    acc = torch.zeros(x.shape[0], out_c, x.shape[2], x.shape[3])
    for head, (i, j) in enumerate(
        (i, j) for i in range(k) for j in range(k)
    ):
        # conv2d correlates, so kernel entry (i, j) multiplies the input at
        # offset (i - radius, j - radius), which is exactly this head's gather.
        acc += torch.einsum("bchw,oc->bohw", gathered[head], weight[:, :, i, j])

    padded = F.pad(x, (radius, radius, radius, radius), mode="circular")
    reference = F.conv2d(padded, weight)
    return (acc - reference).abs().max().item()
