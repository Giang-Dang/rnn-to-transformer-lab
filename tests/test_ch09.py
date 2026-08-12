"""Chapter 9: the closed forms in `scaling.py`, checked against real modules.

A formula about parameter counts is worth nothing until something confirms it
counts *these* parameters. So the two tests that matter here build the modules
this repo builds and compare against `sum(p.numel())`, and the FLOP test
compares against torch's own operator counter rather than against a second
copy of the same arithmetic.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

from rnn_to_transformer_lab.cost import encoder_layer_flops, quadratic_half_point
from rnn_to_transformer_lab.scaling import (
    attention_share,
    block_params,
    decoder_only_params,
    encoder_only_params,
    flops_per_token,
    kaplan_compute,
    six_nd_error,
)
from rnn_to_transformer_lab.transformer import EncoderLayer

SHAPES = ((64, 256, 4), (128, 512, 8), (256, 1024, 8), (768, 3072, 12))


@pytest.mark.parametrize("d_model,d_ff,n_heads", SHAPES)
def test_block_params_matches_a_built_layer(d_model, d_ff, n_heads):
    """The closed form is the count of this repo's own EncoderLayer.

    `bias=False` on the four projections is what `MultiHeadAttention` does,
    so the formula is asked for the same reading.
    """
    layer = EncoderLayer(d_model, n_heads, d_ff)
    built = sum(p.numel() for p in layer.parameters())
    assert block_params(d_model, d_ff, attention_bias=False, ffn_bias=True) == built


@pytest.mark.parametrize("d_model,d_ff,n_heads", SHAPES)
def test_each_bias_switch_is_worth_what_it_should_be(d_model, d_ff, n_heads):
    """Attention biases cost 4*d; feed-forward biases cost d_ff + d.

    Written as two assertions rather than one because the single-flag version
    of `block_params` passed a test like this and still missed a built stack.
    """
    bare = block_params(d_model, d_ff, attention_bias=False, ffn_bias=False)
    attn_only = block_params(d_model, d_ff, attention_bias=True, ffn_bias=False)
    ffn_only = block_params(d_model, d_ff, attention_bias=False, ffn_bias=True)
    assert attn_only - bare == 4 * d_model
    assert ffn_only - bare == d_ff + d_model


def test_a_built_stack_matches_the_decoder_only_formula():
    """Assemble embeddings plus blocks plus a final norm, and count."""
    n_layers, d_model, vocab, n_ctx, d_ff, n_heads = 3, 128, 500, 64, 512, 4
    stack = nn.ModuleDict(
        {
            "tokens": nn.Embedding(vocab, d_model),
            "positions": nn.Embedding(n_ctx, d_model),
            "blocks": nn.ModuleList(
                EncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)
            ),
            "final_norm": nn.LayerNorm(d_model),
        }
    )
    built = sum(p.numel() for p in stack.parameters())
    counted = decoder_only_params(
        n_layers, d_model, vocab, n_ctx, d_ff=d_ff, attention_bias=False
    )
    assert counted.total == built
    assert counted.embedding == (vocab + n_ctx) * d_model
    assert counted.non_embedding == built - counted.embedding


@pytest.mark.parametrize("n_layers,d_model,n_ctx", ((2, 64, 32), (4, 128, 64), (2, 256, 128)))
def test_flops_per_token_against_torchs_own_counter(n_layers, d_model, n_ctx):
    """The per-token count times n is what FlopCounterMode sees for the stack.

    This is chapter 8's cross-check applied to a stack rather than a layer.
    If it fails, the analytic count is wrong and every table chapter 9 prints
    from it is wrong with it.
    """
    d_ff = 4 * d_model
    layers = nn.ModuleList(EncoderLayer(d_model, 4, d_ff) for _ in range(n_layers))
    x = torch.randn(1, n_ctx, d_model)
    with torch.no_grad():
        with FlopCounterMode(display=False) as counter:
            h = x
            for layer in layers:
                h, _ = layer(h, None)
    analytic = n_ctx * flops_per_token(n_layers, d_model, n_ctx, d_ff).total
    assert analytic == counter.get_total_flops()


@pytest.mark.parametrize("n_layers,d_model,n_ctx", ((1, 64, 32), (6, 512, 128), (12, 768, 512)))
def test_per_token_count_agrees_with_chapter_eights_per_layer_count(
    n_layers, d_model, n_ctx
):
    """Two modules, two derivations, one number."""
    d_ff = 4 * d_model
    per_layer = encoder_layer_flops(n_ctx, d_model, d_ff).total
    per_token = flops_per_token(n_layers, d_model, n_ctx, d_ff).total
    assert n_layers * per_layer == n_ctx * per_token


def test_attention_reaches_half_at_chapter_eights_crossover():
    """`n = 2d + d_ff` is where the score matrix costs as much as the rest."""
    for d_model in (64, 512, 768, 12288):
        n = quadratic_half_point(d_model)
        assert attention_share(n, d_model) == pytest.approx(0.5)
        assert n == 6 * d_model


def test_attention_share_depends_only_on_the_ratio():
    """Same n/d, same share, at any width and any depth."""
    assert attention_share(512, 768) == pytest.approx(attention_share(1024, 1536))
    assert attention_share(2048, 12288) == pytest.approx(attention_share(1024, 6144))


def test_six_nd_is_low_and_by_how_much():
    """The approximation errs one way, and the size of the error is n/(6d).

    GPT-3's configuration is the interesting case: at n_ctx 2048 against
    d_model 12288 the error is under three percent, which is why the papers
    can use 6ND and still be believed. At a context of 32768 in the same model
    it is over forty percent, which is why a later chapter cannot.
    """
    gpt3 = decoder_only_params(96, 12288, 50257, 2048)
    tight = six_nd_error(96, 12288, 2048, gpt3.non_embedding)
    loose = six_nd_error(96, 12288, 32768, gpt3.non_embedding)
    assert 1.02 < tight < 1.03
    assert 1.44 < loose < 1.45
    assert tight == pytest.approx(1 + 2048 / (6 * 12288), abs=0.002)


def test_kaplan_compute_is_linear_in_both_arguments():
    n, d = 10**8, 10**10
    assert kaplan_compute(n, d) == 6 * n * d
    assert kaplan_compute(2 * n, d) == 2 * kaplan_compute(n, d)
    assert kaplan_compute(n, 2 * d) == 2 * kaplan_compute(n, d)


def test_embedding_fraction_collapses_with_scale():
    """Why Kaplan's exclusion of the embeddings is not bookkeeping.

    The same table of token embeddings is a quarter of a 117M model and a
    third of a percent of a 175B one, so N and the total parameter count are
    two different quantities that drift apart by two orders of magnitude over
    the range the laws were fitted on.
    """
    small = decoder_only_params(12, 768, 40478, 512)
    large = decoder_only_params(96, 12288, 50257, 2048)
    assert small.embedding_fraction > 0.25
    assert large.embedding_fraction < 0.005
    assert small.embedding_fraction / large.embedding_fraction > 50


def test_rejects_impossible_shapes():
    with pytest.raises(ValueError):
        block_params(0)
    with pytest.raises(ValueError):
        flops_per_token(0, 64, 32)
    with pytest.raises(ValueError):
        flops_per_token(2, 64, -1)
    with pytest.raises(ValueError):
        kaplan_compute(0, 10)
