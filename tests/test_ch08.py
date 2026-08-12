"""Chapter 8: the cost formulas, the two layer-norm orders, and the schedule.

Properties rather than trained numbers, for the reason the chapter 5, 6 and 7
suites give. Two of these matter more than the rest.

`test_analytic_flops_match_torchs_own_counter` is what makes `cost.py` worth
printing at all: the whole first section of the chapter is arithmetic, and
arithmetic nobody checked against a running model is a guess with decimal
places. The counter is instrumentation of the operators that actually ran.

`test_post_ln_is_untouched_by_the_pre_ln_option` is the regression guard for
this tag. Chapter 7's table was measured before `norm_first` existed, and its
row is reproduced inside chapter 8's own tables, so an accidental change to the
default path would silently move a published number in a drafted chapter.
"""

from __future__ import annotations

import math

import torch
from torch.utils.flop_counter import FlopCounterMode

from rnn_to_transformer_lab.cost import (
    attention_beats_lstm_below,
    encoder_layer_flops,
    lstm_layer_flops,
    quadratic_half_point,
    score_matrix_bytes,
)
from rnn_to_transformer_lab.toy_corpus import _pad, sentence, vocabularies
from rnn_to_transformer_lab.transformer import (
    EncoderLayer,
    Transformer,
    warmup_lambda,
)


def build(d_model=32, n_heads=4, n_layers=2, seed=0, **kwargs):
    source_vocab, target_vocab = vocabularies()
    torch.manual_seed(seed)
    model = Transformer(
        source_vocab, target_vocab, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, **kwargs,
    )
    return model, source_vocab, target_vocab


def a_batch(source_vocab, target_vocab, seed=0):
    torch.manual_seed(seed)
    pairs = [sentence(torch.Generator().manual_seed(s)) for s in range(4)]
    source = _pad([source_vocab.encode(p.source) for p in pairs])
    target = _pad([target_vocab.encode(p.target) for p in pairs])
    return source, target


def test_analytic_flops_match_torchs_own_counter():
    """cost.py against instrumentation of the operators that really ran."""
    d_model, d_ff, n_heads = 64, 256, 8
    layer = EncoderLayer(d_model, n_heads, d_ff)
    for n in (8, 32, 128):
        x = torch.randn(1, n, d_model)
        with torch.no_grad():
            with FlopCounterMode(display=False) as counter:
                layer(x, None)
        assert encoder_layer_flops(n, d_model, d_ff).total == counter.get_total_flops()


def test_the_head_count_cancels_out_of_the_flop_formula():
    """Section 3.2.2's "similar computational cost", as an exact equality.

    d_k = d_model / h, and the scores cost 2 n^2 h d_k, so h cancels. The clock
    disagrees and `ch08_clock.py` measures by how much; that disagreement is
    only interesting because this equality is exact.
    """
    counts = {
        heads: encoder_layer_flops(128, 512, 2048).total for heads in (1, 2, 4, 8, 16)
    }
    assert len(set(counts.values())) == 1


def test_the_quadratic_half_point_is_where_the_two_halves_are_equal():
    for d_model, d_ff in ((64, 256), (512, 2048), (256, 1024), (512, 4096)):
        n = quadratic_half_point(d_model, d_ff)
        flops = encoder_layer_flops(n, d_model, d_ff)
        assert flops.quadratic == flops.linear
        assert flops.quadratic_fraction == 0.5
        assert n == 2 * d_model + d_ff


def test_attention_beats_an_lstm_layer_below_twice_the_width():
    """And the paper's own threshold, n < d, is inside that range."""
    for d in (64, 128, 512):
        n = attention_beats_lstm_below(d)
        flops = encoder_layer_flops(n, d)
        attention_only = flops.projections + flops.quadratic
        assert attention_only == lstm_layer_flops(n, d)
        # Strictly cheaper at the paper's own threshold, strictly dearer past
        # this one, so the crossing is real rather than an artifact of rounding.
        below = encoder_layer_flops(d, d)
        assert below.projections + below.quadratic < lstm_layer_flops(d, d)
        above = encoder_layer_flops(2 * n, d)
        assert above.projections + above.quadratic > lstm_layer_flops(2 * n, d)


def test_the_score_matrix_is_square_in_n_and_linear_in_heads():
    assert score_matrix_bytes(2048, 8, batch=8) == 2**30
    assert score_matrix_bytes(200, 4, batch=2) == 2 * 4 * 200 * 200 * 4
    assert score_matrix_bytes(512, 8) == 4 * score_matrix_bytes(256, 8)


def test_post_ln_is_untouched_by_the_pre_ln_option():
    """The default path must be bit-identical to what tag ch07 produced.

    Chapter 7's published table is reproduced as a control row inside chapter
    8's own tables, so if this drifts, two chapters disagree in print.
    """
    model, source_vocab, target_vocab = build()
    source, target = a_batch(source_vocab, target_vocab)
    assert model.norm_first is False
    assert model.label_smoothing == 0.0
    assert isinstance(model.encoder_norm, torch.nn.Identity)
    assert isinstance(model.decoder_norm, torch.nn.Identity)

    reference, _, _ = build()
    assert torch.equal(model.loss(source, target), reference.loss(source, target))


def test_pre_ln_is_a_different_model_and_not_a_relabelled_one():
    post, source_vocab, target_vocab = build(norm_first=False)
    pre, _, _ = build(norm_first=True)
    source, target = a_batch(source_vocab, target_vocab)
    assert not torch.allclose(post.loss(source, target), pre.loss(source, target))


def test_pre_ln_adds_exactly_one_final_layernorm_per_stack():
    """Four times d_model: a weight and a bias, on each of the two stacks.

    Easy to omit, and omitting it costs nothing visible - the model trains,
    the loss falls, and the stack's output is simply never normalized.
    """
    for d_model in (24, 64, 512):
        post, _, _ = build(d_model=d_model, norm_first=False)
        pre, _, _ = build(d_model=d_model, norm_first=True)
        post_count = sum(p.numel() for p in post.parameters())
        pre_count = sum(p.numel() for p in pre.parameters())
        assert pre_count - post_count == 4 * d_model


def test_label_smoothing_raises_the_loss_of_a_confident_model():
    """And leaves an unsmoothed model's loss exactly where it was."""
    plain, source_vocab, target_vocab = build(label_smoothing=0.0)
    smoothed, _, _ = build(label_smoothing=0.1)
    source, target = a_batch(source_vocab, target_vocab)
    assert smoothed.loss(source, target) > plain.loss(source, target)


def test_the_warmup_schedule_peaks_exactly_at_warmup_steps():
    """Equation (3)'s two arms meet at their own switch point.

    The `min` is continuous there, which is the property that makes the
    schedule a triangle joined to a curve rather than a jump.
    """
    d_model, warmup = 512, 4000
    scale = warmup_lambda(d_model, warmup)
    peak = scale(warmup - 1)  # LambdaLR counts from 0, the paper's step_num from 1
    assert peak == d_model ** -0.5 * warmup ** -0.5
    assert scale(warmup - 2) < peak
    assert scale(warmup) < peak
    # Linear on the way up: doubling the step doubles the rate.
    assert math.isclose(scale(199), 2 * scale(99), rel_tol=1e-12)
    # Inverse square root on the way down.
    assert math.isclose(
        scale(4 * warmup - 1), 0.5 * scale(warmup - 1), rel_tol=1e-12
    )


def test_a_wider_model_gets_a_smaller_learning_rate_under_the_paper_recipe():
    """The d_model^-0.5 factor, which is why the number cannot be copied."""
    narrow = warmup_lambda(64, 400)
    wide = warmup_lambda(512, 400)
    assert narrow(399) > wide(399)
    assert math.isclose(narrow(399) / wide(399), math.sqrt(8), rel_tol=1e-12)
