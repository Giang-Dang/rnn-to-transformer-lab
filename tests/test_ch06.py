"""Chapter 6: additive attention, the bidirectional encoder, and the masking.

Properties, not trained numbers, for the reason chapter 5's suite gives: at this
scale an accuracy is a fact about a seed. What is locked here is the structure
the chapter's argument stands on, and in particular the two things that go
wrong silently - the padding guard on the backward pass, and the mask inside
the softmax. Both cost a few points of accuracy and neither raises anything.
"""

from __future__ import annotations

import torch

from rnn_to_transformer_lab.attention import (
    AttentionSeq2Seq,
    greedy_decode,
    train_one,
)
from rnn_to_transformer_lab.toy_corpus import _pad, sentence, vocabularies


def build(n_hidden=16, seed=0, bidirectional=True):
    source_vocab, target_vocab = vocabularies()
    torch.manual_seed(seed)
    model = AttentionSeq2Seq(
        source_vocab, target_vocab, n_hidden=n_hidden,
        bidirectional=bidirectional,
    )
    return model, source_vocab, target_vocab


def test_attention_weights_are_a_distribution_over_source_positions():
    model, source_vocab, _ = build()
    generator = torch.Generator().manual_seed(1)
    tokens = source_vocab.encode(sentence(generator).source)
    source = _pad([tokens])
    annotations, mask, projected, state = model.encode(source)
    alpha = model.attention.weights(state[1], projected, mask)
    assert alpha.shape == (source.shape[0], 1)
    assert torch.allclose(alpha.sum(dim=0), torch.ones(1), atol=1e-6)
    assert bool((alpha >= 0).all())


def test_padding_gets_exactly_zero_attention():
    """The mask, checked where it matters: a short sentence in a long batch.

    Without the masked_fill the softmax runs over the padded columns too, so a
    short sentence spends part of its attention budget on <pad>. It costs a few
    points and nothing reports it.
    """
    model, source_vocab, _ = build()
    generator = torch.Generator().manual_seed(2)
    short = source_vocab.encode(sentence(generator, max_clauses=1).source)
    long = source_vocab.encode(sentence(generator, max_clauses=2).source)
    assert len(long) > len(short)

    source = _pad([short, long])
    annotations, mask, projected, state = model.encode(source)
    alpha = model.attention.weights(state[1], projected, mask)
    assert bool((alpha[len(short) :, 0] == 0).all())
    assert torch.allclose(alpha[:, 0].sum(), torch.ones(()), atol=1e-6)


def test_annotations_ignore_what_a_sentence_was_batched_with():
    """The padding guard, on both directions of the encoder.

    The backward scan is the easy one to get wrong: run it from the batch's
    last column rather than from each sentence's own last token and every short
    sentence gets annotations that begin with a run over <pad>. Same failure
    mode as chapter 5's encoder guard, one direction further.
    """
    model, source_vocab, _ = build()
    generator = torch.Generator().manual_seed(3)
    short = source_vocab.encode(sentence(generator, max_clauses=1).source)
    long = source_vocab.encode(sentence(generator, max_clauses=2).source)
    assert len(long) > len(short)

    alone, _ = model.encoder(
        _pad([short]), model.source_embedding(_pad([short]))
    )
    batched_source = _pad([short, long])
    batched, _ = model.encoder(
        batched_source, model.source_embedding(batched_source)
    )
    assert torch.allclose(
        alone[: len(short), 0], batched[: len(short), 0], atol=1e-6
    )


def test_the_backward_pass_reads_the_words_after_a_position():
    """What makes an annotation bidirectional, asserted rather than assumed.

    Changing a token *after* position j must move annotation j. For a forward
    only encoder it must not, and both halves are checked so that the ablation
    in ch06_encoder.py is known to be the ablation it claims to be.
    """
    generator = torch.Generator().manual_seed(4)
    for bidirectional, should_move in ((True, True), (False, False)):
        model, source_vocab, _ = build(bidirectional=bidirectional)
        tokens = source_vocab.encode(sentence(generator, max_clauses=1).source)
        altered = list(tokens)
        altered[-1] = source_vocab.index["window"]
        assert altered != tokens

        first, _ = model.encoder(_pad([tokens]), model.source_embedding(_pad([tokens])))
        second, _ = model.encoder(
            _pad([altered]), model.source_embedding(_pad([altered]))
        )
        moved = not torch.allclose(first[0, 0], second[0, 0], atol=1e-6)
        assert moved is should_move


def test_precomputing_u_a_h_matches_computing_it_per_step():
    """Appendix A.1.2's one optimisation, checked to be only an optimisation."""
    model, source_vocab, _ = build()
    generator = torch.Generator().manual_seed(5)
    source = _pad([source_vocab.encode(sentence(generator).source)])
    annotations, mask, projected, state = model.encode(source)

    direct = model.attention.u_a(annotations)
    assert torch.allclose(direct, projected, atol=1e-6)


def test_scores_start_at_zero_so_attention_starts_uniform():
    """Appendix B.1 initialises v_a to zero, and ch06_gradient.py rests on it."""
    model, source_vocab, _ = build()
    assert torch.equal(
        model.attention.v_a.weight, torch.zeros_like(model.attention.v_a.weight)
    )
    generator = torch.Generator().manual_seed(6)
    tokens = source_vocab.encode(sentence(generator).source)
    source = _pad([tokens])
    annotations, mask, projected, state = model.encode(source)
    alpha = model.attention.weights(state[1], projected, mask)
    uniform = torch.full_like(alpha, 1.0 / len(tokens))
    assert torch.allclose(alpha, uniform, atol=1e-6)


def test_stepping_token_by_token_matches_teacher_forcing():
    """The same seam chapter 5 locks, now with a context vector per step.

    Training runs `decode_forced`; every table comes from `greedy_decode`
    driving `step` by hand. If the two drift the chapter measures a model it
    did not train.
    """
    model, source_vocab, target_vocab = build()
    generator = torch.Generator().manual_seed(7)
    pair = sentence(generator)
    source = _pad([source_vocab.encode(pair.source)])
    tokens = [target_vocab.index["<sos>"]] + target_vocab.encode(pair.target)
    column = torch.tensor(tokens, dtype=torch.long).unsqueeze(1)

    with torch.no_grad():
        encoded = model.encode(source)
        forced = model.decode_forced(encoded, column)

        annotations, mask, projected, state = model.encode(source)
        stepped = []
        for token in tokens:
            embedded = model.target_embedding(
                torch.tensor([token], dtype=torch.long)
            )
            state, _ = model.step(embedded, state, annotations, mask, projected)
            stepped.append(model.readout(state[1]))

    assert torch.allclose(forced, torch.stack(stepped), atol=1e-6)


def test_the_decoder_sees_one_annotation_per_source_word():
    """Chapter 5's bottleneck assertion, inverted.

    `test_context_width_does_not_depend_on_sentence_length` in the chapter 5
    suite asserts that the decoder always receives 2 * n_hidden numbers. This
    asserts the opposite property for this model: what reaches the attention
    layer grows with the sentence, which is the whole of what changed.
    """
    model, source_vocab, _ = build(n_hidden=32)
    generator = torch.Generator().manual_seed(8)
    seen = set()
    for max_clauses in (1, 2):
        for _ in range(5):
            tokens = source_vocab.encode(sentence(generator, max_clauses).source)
            annotations, _, _, _ = model.encode(_pad([tokens]))
            assert annotations.shape[0] == len(tokens)
            seen.add(annotations.numel())
    assert len(seen) > 1, "the annotation set did not grow with the sentence"


def test_a_trained_model_puts_its_peak_on_a_plausible_word():
    """One epoch is enough to check the plumbing carries a signal at all.

    Not an accuracy assertion: one epoch of this corpus does not translate.
    What it rules out is an alignment matrix that is uniform or degenerate
    after training, which is what a mis-wired score function produces.
    """
    model, _ = train_one(0, n_hidden=16, epochs=1)
    source_vocab, _ = vocabularies()
    generator = torch.Generator().manual_seed(9)
    tokens = source_vocab.encode(sentence(generator, max_clauses=1).source)
    _, weights = greedy_decode(model, _pad([tokens]), keep_weights=True)
    matrix = torch.stack([w[:, 0] for w in weights])
    assert matrix.shape[1] == len(tokens)
    assert float(matrix.max()) > 1.5 / len(tokens)
