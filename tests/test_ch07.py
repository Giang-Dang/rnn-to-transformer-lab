"""Chapter 7: scaling, heads, positions, and the two masks.

Properties rather than trained numbers, for the reason the chapter 5 and 6
suites give: at this scale an accuracy is a fact about a seed. What is locked
here is what the chapter's argument stands on, and four of these tests exist
because the thing they check fails *silently* - a model with no positional
encoding still trains, a decoder that leaks the future still reports a falling
loss, and a mask applied one line too late still produces a distribution that
sums to one.
"""

from __future__ import annotations

import math

import torch

from rnn_to_transformer_lab.toy_corpus import _pad, sentence, vocabularies
from rnn_to_transformer_lab.transformer import (
    MultiHeadAttention,
    PositionwiseFeedForward,
    Transformer,
    causal_mask,
    positional_encoding,
    scaled_dot_product,
)


def build(d_model=32, n_heads=4, n_layers=2, seed=0):
    source_vocab, target_vocab = vocabularies()
    torch.manual_seed(seed)
    model = Transformer(
        source_vocab, target_vocab, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers,
    )
    return model, source_vocab, target_vocab


def test_self_attention_alone_cannot_tell_two_orderings_apart():
    """Why section 3.5 has to exist at all.

    Attention is a weighted sum over a set, so permuting the input permutes
    the output and changes nothing else. A bag of words and a sentence are the
    same object to it. This is asserted rather than argued because the failure
    it prevents is invisible: a model with no positional information still
    trains, still drops its loss, and simply cannot represent word order.
    """
    torch.manual_seed(0)
    layer = MultiHeadAttention(d_model=16, n_heads=4)
    x = torch.randn(1, 6, 16)
    order = torch.tensor([3, 0, 5, 1, 4, 2])

    straight, _ = layer(x, x, x)
    shuffled, _ = layer(x[:, order], x[:, order], x[:, order])
    assert torch.allclose(straight[:, order], shuffled, atol=1e-6)


def test_the_positional_table_breaks_that_symmetry():
    """The other half of the previous test, on the whole encoder."""
    model, source_vocab, _ = build()
    generator = torch.Generator().manual_seed(1)
    tokens = source_vocab.encode(sentence(generator, max_clauses=1).source)
    source = _pad([tokens])
    swapped = source.clone()
    swapped[0, 0], swapped[1, 0] = source[1, 0].clone(), source[0, 0].clone()
    assert not torch.equal(source, swapped)

    with torch.no_grad():
        straight, _ = model.encode(source)
        exchanged, _ = model.encode(swapped)
    assert not torch.allclose(straight[0, 0], exchanged[0, 1], atol=1e-5)


def test_shifting_a_position_is_a_rotation_that_does_not_depend_on_where():
    """Section 3.5's hypothesis, checked as arithmetic rather than believed.

    For a fixed offset k the map from PE(pos) to PE(pos+k) is one 2x2 rotation
    per frequency pair, by an angle that depends on k and on the pair but not
    on pos. That independence is the whole claim: it is what makes the shift a
    single linear map rather than a different map at every position.
    """
    d_model, k = 32, 5
    table = positional_encoding(60, d_model)
    for pair in range(0, d_model, 2):
        omega = 1.0 / math.pow(10000.0, pair / d_model)
        angle = k * omega
        rotation = torch.tensor(
            [[math.cos(angle), math.sin(angle)],
             [-math.sin(angle), math.cos(angle)]]
        )
        for pos in (0, 1, 7, 30, 54):
            source = table[pos, pair : pair + 2]
            assert torch.allclose(
                rotation @ source, table[pos + k, pair : pair + 2], atol=1e-5
            )


def test_the_causal_mask_lets_no_position_see_a_later_token():
    """Section 3.2.3, tested by intervention rather than by reading the mask.

    Change the token at position j and every output before j must be
    bit-identical. Checking the mask tensor itself would only confirm the
    tensor; this confirms that the tensor is wired to the softmax that uses it.
    """
    model, source_vocab, target_vocab = build()
    generator = torch.Generator().manual_seed(2)
    pair = sentence(generator, max_clauses=1)
    source = _pad([source_vocab.encode(pair.source)])
    tokens = [target_vocab.index["<sos>"]] + target_vocab.encode(pair.target)
    column = torch.tensor(tokens, dtype=torch.long).unsqueeze(1)

    altered = column.clone()
    cut = len(tokens) // 2
    altered[cut, 0] = target_vocab.index["mèo"]
    assert not torch.equal(column, altered)

    with torch.no_grad():
        memory, mask = model.encode(source)
        straight, _ = model.decode(memory, mask, column)
        changed, _ = model.decode(memory, mask, altered)

    assert torch.equal(straight[:cut], changed[:cut])
    assert not torch.allclose(straight[cut:], changed[cut:], atol=1e-6)


def test_masking_after_the_softmax_is_the_same_function_until_it_underflows():
    """Section 3.2.3 says "in the input of the softmax", and the reason is
    numerical rather than mathematical.

    Both halves are asserted here because the first is the one people get
    wrong in argument and the second is the one that bites in practice.

    Restricting a softmax to a subset and renormalising gives back exactly the
    softmax over that subset - the shared normaliser cancels - so on ordinary
    scores the two orders agree to floating point. What does not survive is a
    row where the kept scores sit far below the masked ones: the kept mass
    underflows to zero before anything renormalises it, and 0/0 is NaN. Doing
    it inside never forms that ratio.
    """
    torch.manual_seed(3)
    query = torch.randn(1, 1, 4, 8)
    key = torch.randn(1, 1, 4, 8)
    value = torch.randn(1, 1, 4, 8)
    mask = causal_mask(4)

    _, inside = scaled_dot_product(query, key, value, mask)
    outside = torch.softmax(query @ key.transpose(-2, -1) / math.sqrt(8), dim=-1)
    outside = outside * mask
    outside = outside / outside.sum(dim=-1, keepdim=True)
    assert torch.allclose(inside, outside, atol=1e-7)

    # The same two orders on a row the first cannot represent.
    scores = torch.tensor([[[[900.0, -900.0]]]])
    keep = torch.tensor([[[[False, True]]]])
    inside = torch.softmax(scores.masked_fill(~keep, float("-inf")), dim=-1)
    outside = torch.softmax(scores, dim=-1) * keep
    outside = outside / outside.sum(dim=-1, keepdim=True)

    assert torch.allclose(inside, torch.tensor([[[[0.0, 1.0]]]]))
    assert bool(torch.isnan(outside).any())


def test_padding_gets_exactly_zero_attention():
    """Chapter 6's silent failure, one architecture later.

    Same defect and the same absence of any signal: a short sentence sharing a
    batch with a long one spends attention on <pad> and only the accuracy
    notices.
    """
    model, source_vocab, _ = build()
    generator = torch.Generator().manual_seed(4)
    short = source_vocab.encode(sentence(generator, max_clauses=1).source)
    long = source_vocab.encode(sentence(generator, max_clauses=2).source)
    assert len(long) > len(short)

    source = _pad([short, long])
    tokens = source.transpose(0, 1)
    x = model._embed(tokens, model.source_embedding)
    from rnn_to_transformer_lab.transformer import padding_mask

    _, weights = model.encoder_layers[0].self_attention(
        x, x, x, padding_mask(tokens)
    )
    assert bool((weights[0, :, :, len(short) :] == 0).all())
    assert torch.allclose(
        weights[0].sum(dim=-1), torch.ones_like(weights[0].sum(dim=-1)), atol=1e-6
    )


def test_the_head_count_does_not_move_the_parameter_count():
    """Section 3.2.2's accounting claim, which is exactly why d_k = d_model / h.

    Splitting into more heads narrows each one by the same factor, so the four
    projections keep their shape. A reader who expects eight heads to cost
    eight times as much is reading the mechanism the way it is usually drawn
    rather than the way it is defined.
    """
    counts = set()
    for n_heads in (1, 2, 4, 8, 16):
        layer = MultiHeadAttention(d_model=64, n_heads=n_heads)
        counts.add(sum(p.numel() for p in layer.parameters()))
    assert len(counts) == 1, counts


def test_the_feed_forward_network_moves_nothing_between_positions():
    """What "position-wise" means, asserted.

    In this architecture attention is the only thing that moves information
    sideways. If the feed-forward network did too, every statement the chapter
    makes about where mixing happens would be wrong.
    """
    torch.manual_seed(5)
    network = PositionwiseFeedForward(d_model=16, d_ff=32)
    x = torch.randn(1, 5, 16)
    altered = x.clone()
    altered[0, 2] = torch.randn(16)

    with torch.no_grad():
        straight = network(x)
        changed = network(altered)

    moved = [
        not torch.allclose(straight[0, i], changed[0, i], atol=1e-7)
        for i in range(5)
    ]
    assert moved == [False, False, True, False, False]


def test_scaling_keeps_the_softmax_out_of_saturation():
    """The property behind the derivation, at two widths far enough apart.

    Not the measurement - `experiments/ch07_scaling.py` is where the numbers
    come from. This asserts only the direction, so that a change which removes
    the division fails here rather than in a chapter's table.
    """
    torch.manual_seed(6)
    peaks = {}
    for d_k in (8, 512):
        query = torch.randn(1, 1, 1, d_k)
        key = torch.randn(1, 1, 64, d_k)
        raw = (query @ key.transpose(-2, -1)).squeeze()
        peaks[d_k] = (
            float(torch.softmax(raw, dim=-1).max()),
            float(torch.softmax(raw / math.sqrt(d_k), dim=-1).max()),
        )
    assert peaks[512][0] > peaks[8][0]
    assert peaks[512][1] < peaks[512][0]
    assert abs(peaks[512][1] - peaks[8][1]) < 0.2


def test_decoding_step_by_step_matches_teacher_forcing():
    """The seam chapters 5 and 6 both lock, in the shape this model needs.

    Training runs the decoder once over the whole target; every table comes
    from `greedy_decode`, which re-runs it over a growing prefix. The causal
    mask is what makes those two agree, so this test is also the strongest
    statement that the mask is doing its job.
    """
    model, source_vocab, target_vocab = build()
    generator = torch.Generator().manual_seed(7)
    pair = sentence(generator)
    source = _pad([source_vocab.encode(pair.source)])
    tokens = [target_vocab.index["<sos>"]] + target_vocab.encode(pair.target)
    column = torch.tensor(tokens, dtype=torch.long).unsqueeze(1)

    with torch.no_grad():
        memory, mask = model.encode(source)
        forced, _ = model.decode(memory, mask, column)
        for cut in range(1, len(tokens) + 1):
            prefix, _ = model.decode(memory, mask, column[:cut])
            assert torch.allclose(prefix[cut - 1], forced[cut - 1], atol=1e-6)
