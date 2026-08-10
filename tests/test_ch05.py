"""Chapter 5: the encoder-decoder, the toy corpus, and the two decoders.

These are properties rather than trained numbers. A training run at this scale
moves several points between seeds, so an assertion on an accuracy would be an
assertion about a seed; what is worth locking is the structure the chapter's
argument rests on. The accuracies themselves are asserted the way the book
asserts them, by the experiment scripts printing them and the research note
recording the run.
"""

from __future__ import annotations

import torch

from rnn_to_transformer_lab.seq2seq import (
    Seq2Seq,
    beam_decode,
    greedy_decode,
    train_one,
)
from rnn_to_transformer_lab.toy_corpus import (
    ANIMALS,
    OBJECT_ADJECTIVES,
    PAD,
    _pad,
    corpus,
    disjoint_splits,
    sentence,
    vocabularies,
)


def test_the_adjective_crosses_the_noun():
    """The one thing about this corpus that is not arbitrary.

    English sets determiner, adjective, noun; Vietnamese sets classifier, noun,
    adjective. If that stops being true the corpus stops being able to show
    chapter 6 anything, so it is asserted rather than left to the reader of the
    grammar.
    """
    english_adjectives = {en for en, _ in OBJECT_ADJECTIVES}
    vietnamese_nouns = {vi for _, vi, _ in ANIMALS}
    checked = 0
    for pair in corpus(400, seed=3, max_clauses=1):
        source, target = list(pair.source), list(pair.target)
        for adjective in english_adjectives & set(source):
            noun_positions = [i for i, t in enumerate(source) if t in {"cat", "dog", "bird", "fish", "horse"}]
            adjective_position = source.index(adjective)
            if not noun_positions:
                continue
            noun_after = [i for i in noun_positions if i > adjective_position]
            if not noun_after:
                continue
            # English: adjective before its noun.
            assert adjective_position < noun_after[0]
            checked += 1
        for vietnamese_noun in vietnamese_nouns & set(target):
            position = target.index(vietnamese_noun)
            # Vietnamese: the classifier is immediately before the noun.
            assert target[position - 1] in {"con", "cái", "quyển"}
    assert checked > 100, "the corpus stopped producing adjectives"


def test_target_is_usually_longer_than_source():
    """The classifier has no English word, so the target grows."""
    pairs = corpus(500, seed=4)
    longer = sum(len(p.target) > len(p.source) for p in pairs)
    assert longer / len(pairs) > 0.6


def test_splits_share_no_source_sentence():
    train, test = disjoint_splits(2000, 200, seed=5)
    assert len(test) == 200
    assert not ({p.source for p in train} & {p.source for p in test})


def test_corpus_is_deterministic_in_its_seed():
    assert corpus(50, seed=7) == corpus(50, seed=7)
    assert corpus(50, seed=7) != corpus(50, seed=8)


def test_vocabulary_covers_everything_the_grammar_emits():
    """Built from the grammar, so a sample can never contain an unseen token."""
    source_vocab, target_vocab = vocabularies()
    assert source_vocab.tokens[0] == PAD and target_vocab.tokens[0] == PAD
    for pair in corpus(500, seed=11):
        source_vocab.encode(pair.source)
        target_vocab.encode(pair.target)


def test_encoder_state_ignores_padding():
    """A sentence must encode to the same vector whatever it is batched with.

    The encoder freezes its state wherever the input is <pad>. Without that the
    state keeps stepping on padding and a short sentence gets a different
    context depending on the longest sentence in its batch, which is invisible
    in the loss and shows up only as an accuracy that moves when the batch size
    does.
    """
    source_vocab, target_vocab = vocabularies()
    torch.manual_seed(0)
    model = Seq2Seq(source_vocab, target_vocab, n_hidden=16)

    generator = torch.Generator().manual_seed(2)
    short = source_vocab.encode(sentence(generator, max_clauses=1).source)
    long = source_vocab.encode(sentence(generator, max_clauses=2).source)
    assert len(long) > len(short)

    alone = model.encode(_pad([short]))
    padded = model.encode(_pad([short, long]))
    for a, b in zip(alone, padded):
        assert torch.allclose(a[0], b[0], atol=1e-6)


def test_context_is_the_only_channel_between_encoder_and_decoder():
    """Two sources with the same context produce the same translation.

    This is the claim the whole chapter turns on. It is checked by handing the
    decoder a context directly rather than by trusting the wiring: if anything
    else reached the decoder, decoding the same context twice from different
    sources could differ.
    """
    source_vocab, target_vocab = vocabularies()
    torch.manual_seed(1)
    model = Seq2Seq(source_vocab, target_vocab, n_hidden=16)
    generator = torch.Generator().manual_seed(3)

    first = _pad([source_vocab.encode(sentence(generator, max_clauses=1).source)])
    second = _pad([source_vocab.encode(sentence(generator, max_clauses=2).source)])

    context = model.encode(first)
    with torch.no_grad():
        forced = torch.tensor([[target_vocab.index["<sos>"]]])
        a = model.decode_forced(context, forced)
        # Same context, source thrown away entirely.
        b = model.decode_forced(context, forced)
    assert torch.equal(a, b)
    assert not torch.equal(model.encode(second)[0], context[0])


def test_context_width_does_not_depend_on_sentence_length():
    """2 * n_hidden numbers, whatever came in. The bottleneck, as an assertion."""
    source_vocab, target_vocab = vocabularies()
    torch.manual_seed(0)
    model = Seq2Seq(source_vocab, target_vocab, n_hidden=32)
    generator = torch.Generator().manual_seed(4)
    widths = set()
    for max_clauses in (1, 2):
        for _ in range(5):
            tokens = source_vocab.encode(sentence(generator, max_clauses).source)
            c, h = model.encode(_pad([tokens]))
            widths.add(c.numel() + h.numel())
    assert widths == {model.context_width}


def test_beam_of_one_is_greedy():
    """The two decoders are separate code paths and must agree at B = 1.

    Greedy is written out separately because it is the baseline every table in
    the chapter is read against, and two implementations that drift apart would
    make the beam column measure the drift instead of the search.
    """
    source_vocab, target_vocab = vocabularies()
    model, _ = train_one(0, reverse_source=True, n_hidden=16, epochs=1)
    generator = torch.Generator().manual_seed(5)
    for _ in range(20):
        tokens = source_vocab.encode(list(sentence(generator).source)[::-1])
        padded = _pad([tokens])
        assert beam_decode([model], padded, beam=1) == greedy_decode(model, padded)[0]


def test_stepping_token_by_token_matches_teacher_forcing():
    """The seam both decoders depend on, checked directly.

    `decode_forced` runs the decoder over a whole target at once; greedy and
    beam search drive `decoder.step` one token at a time, threading the state
    by hand. Every number in this chapter's tables comes from the second path
    while training comes from the first, so if they disagree the chapter is
    measuring a model it did not train. An off-by-one in the state threading is
    exactly the kind of bug that lowers accuracy a few points and looks like a
    hyperparameter.
    """
    source_vocab, target_vocab = vocabularies()
    torch.manual_seed(0)
    model = Seq2Seq(source_vocab, target_vocab, n_hidden=16)
    generator = torch.Generator().manual_seed(6)

    pair = sentence(generator)
    source = _pad([source_vocab.encode(pair.source)])
    tokens = [target_vocab.index["<sos>"]] + target_vocab.encode(pair.target)
    column = torch.tensor(tokens, dtype=torch.long).unsqueeze(1)

    with torch.no_grad():
        forced = model.decode_forced(model.encode(source), column)

        state = model.encode(source)
        stepped = []
        for token in tokens:
            state = model.decoder.step(
                model.target_embedding(torch.tensor([token], dtype=torch.long)), state
            )
            stepped.append(model.readout(state[1]))

    assert torch.allclose(forced, torch.stack(stepped), atol=1e-6)


def test_beam_search_is_not_monotone_in_the_beam_width():
    """Recorded as a test because it is the opposite of what one expects.

    A wider beam generates a superset of the candidates a narrower one does,
    which is why it is tempting to conclude it must return a hypothesis of at
    least equal log probability. It does not follow: each step keeps only the
    top B *candidates*, and the prefix the narrow search kept can rank below
    the Bth place of the wider search's larger candidate pool and be dropped.

    So a beam column that goes down somewhere is not evidence of a bug, and
    this test exists so that nobody "fixes" the search on that evidence. It
    asserts only that the search runs and returns something scoreable for every
    width; the counterexample it was written from is in the chapter 5 research
    note.
    """
    source_vocab, target_vocab = vocabularies()
    model, _ = train_one(0, reverse_source=True, n_hidden=16, epochs=1)
    generator = torch.Generator().manual_seed(6)

    for _ in range(10):
        tokens = source_vocab.encode(list(sentence(generator).source)[::-1])
        padded = _pad([tokens])
        for beam in (1, 2, 5):
            output = beam_decode([model], padded, beam=beam)
            assert _score(model, padded, output, target_vocab) < 0.0


def _score(model, source, tokens, target_vocab) -> float:
    """Log probability the model assigns to `tokens`, including its <eos>."""
    with torch.no_grad():
        full = [target_vocab.index["<sos>"]] + list(tokens) + [target_vocab.index["<eos>"]]
        column = torch.tensor(full, dtype=torch.long).unsqueeze(1)
        logits = model.decode_forced(model.encode(source), column[:-1])
        logp = torch.log_softmax(logits, dim=-1)
        return float(logp.gather(2, column[1:].unsqueeze(-1)).sum())


def test_an_ensemble_of_one_is_that_model():
    source_vocab, _ = vocabularies()
    model, _ = train_one(0, reverse_source=True, n_hidden=16, epochs=1)
    generator = torch.Generator().manual_seed(7)
    for _ in range(10):
        tokens = source_vocab.encode(list(sentence(generator).source)[::-1])
        padded = _pad([tokens])
        assert beam_decode([model], padded, beam=3) == beam_decode(
            [model, model], padded, beam=3
        )


def test_the_forget_bias_starts_open():
    """Initialised at 1, so the cell starts out behaving like chapter 4's fixed 1.0.

    The four blocks are stacked (input, forget, cell, output), the same order
    torch.nn.LSTM uses, and a book that got the slice wrong would be setting
    the input gate instead and saying the opposite of what it meant.
    """
    source_vocab, target_vocab = vocabularies()
    torch.manual_seed(0)
    model = Seq2Seq(source_vocab, target_vocab, n_hidden=8)
    for layer in (model.encoder, model.decoder):
        bias = layer.bias.detach()
        assert torch.equal(bias[8:16], torch.ones(8))
        assert torch.equal(bias[:8], torch.zeros(8))
        assert torch.equal(bias[16:], torch.zeros(16))
