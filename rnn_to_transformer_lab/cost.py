"""Table 1 of Vaswani et al. (2017), turned from asymptotics into numbers.

The paper's table 1 gives three columns for four layer types, and the row
chapter 8 is about is the first one: self-attention costs O(n^2 . d) per layer
against a recurrent layer's O(n . d^2). The paper then dismisses the quadratic
term in one sentence of section 4, and the dismissal is a conditional:

    "self-attention layers are faster than recurrent layers when the sequence
    length n is smaller than the representation dimensionality d, which is most
    often the case with sentence representations used by state-of-the-art
    models in machine translations"

Big-O drops the constants, and the constants are the whole question of whether
n < d is the right threshold. This module puts them back. Everything here is
closed form: no model is built and nothing is timed, so these functions give
the same answer on any machine, and `experiments/ch08_flops.py` checks them
against `torch.utils.flop_counter.FlopCounterMode` on a real layer.

**The counting convention.** One multiply-add is 2 FLOPs, so a matrix product
(m x k) by (k x p) costs 2*m*k*p. That is the convention `FlopCounterMode`
uses, which is what makes the cross-check meaningful; halve everything for the
"multiply-accumulate" convention some papers use instead.

**What is deliberately not counted**, and it is worth naming because leaving it
silent would make the numbers look more authoritative than they are: softmax,
layer normalization, the residual additions, ReLU, and the bias adds. Those are
elementwise and cost O(n^2 * h) and O(n * d) respectively rather than a matrix
product, so they vanish beside the terms here at any size worth discussing -
but they do *not* vanish on a clock, because an elementwise pass over the n x n
score matrix touches the same memory a matrix multiply does without the
arithmetic intensity to pay for it. Chapter 8 measures the clock separately for
exactly that reason, and the gap between this file and that measurement is one
of the chapter's points rather than an embarrassment.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LayerFlops:
    """One encoder layer's forward matrix-multiply FLOPs, split by where.

    `scores` and `weighted_values` are the two products that carry an n^2; the
    rest scale linearly in n. Keeping them apart is the point of the class,
    because the whole argument of chapter 8's first section is about which
    group grows faster and where they cross.
    """

    projections: int
    scores: int
    weighted_values: int
    feed_forward: int

    @property
    def quadratic(self) -> int:
        """The part that grows as n^2."""
        return self.scores + self.weighted_values

    @property
    def linear(self) -> int:
        """The part that grows as n * d^2."""
        return self.projections + self.feed_forward

    @property
    def total(self) -> int:
        return self.linear + self.quadratic

    @property
    def quadratic_fraction(self) -> float:
        return self.quadratic / self.total


def encoder_layer_flops(n: int, d_model: int, d_ff: int | None = None) -> LayerFlops:
    """Forward FLOPs of one post-LN encoder layer at batch 1, sequence `n`.

    Term by term, against `transformer.py`:

    * **projections**, `8 * n * d^2`. Four bias-free d x d matrices, W^Q, W^K,
      W^V and W^O, each costing 2*n*d*d.
    * **scores**, `2 * n^2 * d`. Per head, Q K^T is (n x d_k) by (d_k x n) for
      2*n*d_k*n; summed over h heads that is 2*n^2*(h*d_k), and h*d_k is d_model
      exactly, by section 3.2.2's d_k = d_model/h. **So the head count cancels**
      and this number does not depend on h at all - which is the FLOP half of
      the paper's "the total computational cost is similar to that of
      single-head attention", and chapter 8 measures the clock half.
    * **weighted_values**, `2 * n^2 * d`, the same shape one product later.
    * **feed_forward**, `4 * n * d * d_ff`, two matrices of d x d_ff and
      d_ff x d.

    `d_ff` defaults to 4 * d_model, the paper's ratio (2048 against 512) and
    this repo's default.
    """
    if n < 0 or d_model <= 0:
        raise ValueError(f"need n >= 0 and d_model > 0, got n={n}, d_model={d_model}")
    d_ff = 4 * d_model if d_ff is None else d_ff
    return LayerFlops(
        projections=8 * n * d_model * d_model,
        scores=2 * n * n * d_model,
        weighted_values=2 * n * n * d_model,
        feed_forward=4 * n * d_model * d_ff,
    )


def lstm_layer_flops(n: int, d: int) -> int:
    """Forward FLOPs of one LSTM layer, `16 * n * d^2`.

    `seq2seq.LstmLayer` holds `w_x` of shape (4d, d) and `w_h` of shape (4d, d),
    the four gate blocks stacked, which is also how `torch.nn.LSTM` stacks
    them. Each step does both products: 2*d*4d twice, so 16*d^2 per step.

    The factor of 4 is the gates and it is why "a recurrent layer" is not one
    number: a plain RNN layer, one matrix each way, is `4 * n * d^2`, four times
    cheaper. The paper's table 1 writes O(n . d^2) for the whole family and both
    of these live inside that.
    """
    return 16 * n * d * d


def rnn_layer_flops(n: int, d: int) -> int:
    """Forward FLOPs of one plain recurrent layer, `4 * n * d^2`."""
    return 4 * n * d * d


def quadratic_half_point(d_model: int, d_ff: int | None = None) -> int:
    """The `n` at which the n^2 terms reach half of one encoder layer's FLOPs.

    Set quadratic equal to linear and the d_model on both sides cancels:

        4 n^2 d = 8 n d^2 + 4 n d d_ff
              n = 2 d + d_ff

    Exact, and an integer whenever d_ff is. At the paper's base configuration,
    d_model 512 and d_ff 2048, that is **3072** - six times the width the paper
    was dismissing the quadratic term at, and roughly a hundred times the
    sentence lengths it trained on. The reason it sits so far out is the
    feed-forward network: at d_ff = 4 d it contributes 16 n d^2 against the
    projections' 8 n d^2, so two thirds of what the quadratic term has to
    overtake is not attention at all.
    """
    d_ff = 4 * d_model if d_ff is None else d_ff
    return 2 * d_model + d_ff


def attention_beats_lstm_below(d: int) -> int:
    """The `n` below which one self-attention sub-layer costs fewer FLOPs than
    one LSTM layer of the same width.

    The attention sub-layer alone, without the feed-forward network, is
    8 n d^2 + 4 n^2 d. Against `16 n d^2`:

        4 n^2 d = 8 n d^2   =>   n = 2 d

    So with the constants of this repo's two layers in place, the paper's
    "n smaller than d" is off by a factor of two in self-attention's favour.
    Against a *plain* recurrent layer at 4 n d^2 the comparison reverses and
    attention never wins on FLOPs at any n, because its four projections alone
    already cost twice what the whole plain layer does. Which of those two the
    reader has in mind is the difference between the paper's claim being
    conservative and it being false, and the paper says only "recurrent".
    """
    return 2 * d


def score_matrix_bytes(
    n: int, n_heads: int, batch: int = 1, bytes_per_element: int = 4
) -> int:
    """Bytes held by one layer's attention score tensor, `batch*h*n*n*4`.

    This is the tensor that has no equivalent in a recurrent layer at all: a
    recurrent layer's activations are (batch, n, d) and grow linearly, while
    this one is square in n and carries a factor of the head count on top. It
    is also not one tensor per model but one per attention sub-layer, and
    training keeps them all alive for the backward pass.

    Chapter 8 prints this rather than a measured peak, and says why: on CPU the
    score tensor is allocated by libtorch rather than by Python, so
    `tracemalloc` cannot see it and reports a few hundred bytes at every n, and
    process RSS is confounded by the allocator holding freed blocks instead of
    returning them. The arithmetic is exact and the measurement is not, so the
    arithmetic is what goes on the page.
    """
    return batch * n_heads * n * n * bytes_per_element
