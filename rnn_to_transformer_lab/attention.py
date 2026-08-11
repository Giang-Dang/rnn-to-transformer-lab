"""Additive attention, the model of Bahdanau, Cho and Bengio (2015).

Chapter 5's encoder-decoder squeezes a whole source sentence through one
fixed-width vector and then never looks at the source again. This module keeps
every encoder state instead, and lets the decoder build a *different* context
vector at every step by taking a weighted average of them. The paper's
equations (5) and (6):

    c_i     = sum_j alpha_ij h_j
    alpha_ij = softmax_j(e_ij),  e_ij = a(s_{i-1}, h_j)

and appendix A.1.2 fixes the alignment model `a` as a one-hidden-layer network:

    a(s_{i-1}, h_j) = v_a^T tanh(W_a s_{i-1} + U_a h_j)

Four things here follow the paper deliberately, and two depart from it.

**The score is read from `s_{i-1}`, the state *before* the step.** Section 3.1:
the score "is based on the RNN hidden state s_{i-1} (just before emitting y_i)
and the j-th annotation h_j". So the order inside one decoder step is: score
against the old state, build c_i, then step. Luong et al. (2015) reverse this,
computing the context from the new state, and their section 3.1 names the
difference; chapter 6's bridge box is where that is argued.

**`U_a h_j` is precomputed once per sentence.** Appendix A.1.2 says so in as
many words: "Since U_a h_j does not depend on i, we can pre-compute it in
advance to minimize the computational cost." It is the only optimisation the
paper spells out, and `precompute` is where it lives.

**The encoder is bidirectional and the annotation is a concatenation.**
Section 3.2, equation (7): h_j = [forward_j ; backward_j], so an annotation
summarises the words on both sides of x_j rather than only those before it.

**The decoder's initial state comes from the backward encoder.** Appendix
A.2.2: s_0 = tanh(W_s backward_h_1). Not the forward final state, which is what
chapter 5 uses: the backward RNN's state at source position 1 is the one that
has just read the whole sentence right to left.

**The departure: LSTM units, not the gated hidden unit.** The paper uses Cho's
gated unit throughout (appendix A.1.1) and this module uses chapter 5's
`LstmLayer`. The paper licenses exactly this swap in the same appendix: "It is
therefore possible to use LSTM units instead of the gated hidden unit described
here, as was done in a similar context by Sutskever et al. (2014)." Keeping the
cell identical to chapter 5's is what lets chapter 6 attribute a difference in
the tables to attention rather than to the recurrence.

**The departure: a plain readout, not a deep output with maxout.** Appendix
A.2.2 puts a maxout layer between the state and the softmax. This module reads
the softmax straight off s_i, as chapter 5 does, for the same reason: the two
chapters' tables have to be comparable, and the maxout layer is not what either
of them is measuring.
"""

from __future__ import annotations

import torch
from torch import nn

from .seq2seq import PAD_INDEX, LstmLayer
from .toy_corpus import Vocabulary


class BiEncoder(nn.Module):
    """Two LSTMs over the source, one each way; annotations are the concatenation.

    Both directions freeze their state wherever the input is padding, the same
    guard chapter 5's `Seq2Seq.encode` carries. Running backward that way is
    what makes the backward pass start at each sentence's own last real token
    rather than at the batch's last column, and getting it wrong is invisible
    in the loss.
    """

    def __init__(
        self, n_embedding: int, n_hidden: int, bidirectional: bool = True
    ) -> None:
        super().__init__()
        self.n_hidden = n_hidden
        self.bidirectional = bidirectional
        self.ahead = LstmLayer(n_embedding, n_hidden)
        self.back = LstmLayer(n_embedding, n_hidden) if bidirectional else None

    @property
    def n_annotation(self) -> int:
        """Width of one annotation: forward and backward concatenated."""
        return 2 * self.n_hidden if self.bidirectional else self.n_hidden

    def forward(
        self, source: torch.Tensor, embedded: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(steps, batch) indices and their embeddings in; annotations and mask out.

        Returns annotations of shape (steps, batch, n_annotation) and a boolean
        mask of shape (steps, batch) that is True at a real token.

        `bidirectional=False` keeps everything else about the model and drops
        only the backward pass, so an annotation summarises the words up to
        x_j and not the words after it. That is the ablation chapter 6 runs
        against section 3.2's claim, and it is not the same model as chapter
        5's: the decoder still gets one annotation per source position and
        still weights them. Only the right-hand context is gone.
        """
        steps = source.shape[0]
        live = (source != PAD_INDEX).unsqueeze(-1).to(embedded.dtype)
        mask = source != PAD_INDEX

        ahead = self._scan(self.ahead, embedded, live, range(steps))
        if self.back is None:
            return torch.stack(ahead), mask

        backwards = reversed(range(steps))
        back = self._scan(self.back, embedded, live, backwards)
        stacked = [torch.stack(ahead), torch.stack(back)]
        return torch.cat(stacked, dim=-1), mask

    @staticmethod
    def _scan(layer: LstmLayer, embedded, live, order) -> list[torch.Tensor]:
        """One direction. `order` decides which; the result is always in source order."""
        projected = layer.project(embedded)
        state = layer.initial_state(embedded.shape[1], embedded.dtype)
        out: dict[int, torch.Tensor] = {}
        for t in order:
            c_next, h_next = layer.step_projected(projected[t], state)
            state = (
                live[t] * c_next + (1.0 - live[t]) * state[0],
                live[t] * h_next + (1.0 - live[t]) * state[1],
            )
            out[t] = state[1]
        return [out[t] for t in range(embedded.shape[0])]

    def summary(self, annotations: torch.Tensor) -> torch.Tensor:
        """The state appendix A.2.2 feeds to s_0: the backward one at position 1.

        That is the backward RNN's *final* state, since it reads right to left,
        so it is the half that has just seen the whole sentence. Chapter 5 uses
        the forward final state for the same job, and the difference is only
        which end the reading started from.

        Without a backward pass there is no such half, so the ablation falls
        back to the forward state at the last real token. Because both scans
        freeze their state at padding, that state is simply the last row: after
        a sentence ends, every later row repeats it.
        """
        if self.back is None:
            return annotations[-1]
        return annotations[0, :, self.n_hidden :]


class AdditiveAttention(nn.Module):
    """e_ij = v_a^T tanh(W_a s_{i-1} + U_a h_j), softmaxed over j.

    `n_align` is the paper's n', the width of the alignment model's one hidden
    layer. The initialisation is appendix B.1: W_a and U_a from N(0, 0.001^2)
    and v_a exactly zero. Starting v_a at zero makes every score zero on the
    first batch, so the model begins by averaging the annotations uniformly and
    has to learn to prefer any of them. Chapter 6 measures the gradient at that
    point, which is why the detail matters here rather than being tidied away.
    """

    def __init__(self, n_hidden: int, n_annotation: int, n_align: int) -> None:
        super().__init__()
        self.w_a = nn.Linear(n_hidden, n_align, bias=False)
        self.u_a = nn.Linear(n_annotation, n_align, bias=False)
        self.v_a = nn.Linear(n_align, 1, bias=False)
        nn.init.normal_(self.w_a.weight, std=0.001)
        nn.init.normal_(self.u_a.weight, std=0.001)
        nn.init.zeros_(self.v_a.weight)

    def precompute(self, annotations: torch.Tensor) -> torch.Tensor:
        """U_a h_j for every j, which no target step changes."""
        return self.u_a(annotations)

    def weights(
        self,
        state: torch.Tensor,
        projected: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """alpha_ij for one target step: (steps, batch), summing to 1 down dim 0."""
        hidden = torch.tanh(self.w_a(state) + projected)
        scores = self.v_a(hidden).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        return torch.softmax(scores, dim=0)

    def context(
        self, weights: torch.Tensor, annotations: torch.Tensor
    ) -> torch.Tensor:
        """Equation (5): the weighted sum of annotations, (batch, n_annotation)."""
        return (weights.unsqueeze(-1) * annotations).sum(dim=0)


class AttentionSeq2Seq(nn.Module):
    """Chapter 5's model with the fixed-width vector replaced by equation (5).

    The two models are deliberately the same everywhere else: same cell, same
    embedding width, same readout, same corpus, same recipe. `n_hidden` still
    sets the decoder's state width, but it no longer sets how much of the
    source the decoder can see, and that is the whole of chapter 6's claim.
    """

    def __init__(
        self,
        source_vocab: Vocabulary,
        target_vocab: Vocabulary,
        n_hidden: int,
        n_embedding: int = 32,
        n_align: int | None = None,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.n_hidden = n_hidden
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.source_embedding = nn.Embedding(
            len(source_vocab), n_embedding, padding_idx=PAD_INDEX
        )
        self.target_embedding = nn.Embedding(
            len(target_vocab), n_embedding, padding_idx=PAD_INDEX
        )
        self.encoder = BiEncoder(n_embedding, n_hidden, bidirectional=bidirectional)
        self.attention = AdditiveAttention(
            n_hidden, self.encoder.n_annotation, n_align or n_hidden
        )
        # The decoder step reads the previous target word and the context
        # together: s_i = f(s_{i-1}, y_{i-1}, c_i) of equation (4).
        self.decoder = LstmLayer(n_embedding + self.encoder.n_annotation, n_hidden)
        self.bridge = nn.Linear(n_hidden, n_hidden)
        self.readout = nn.Linear(n_hidden, len(target_vocab))

    def encode(self, source: torch.Tensor):
        """Annotations, mask, precomputed U_a h_j, and the initial decoder state."""
        embedded = self.source_embedding(source)
        annotations, mask = self.encoder(source, embedded)
        projected = self.attention.precompute(annotations)
        h_0 = torch.tanh(self.bridge(self.encoder.summary(annotations)))
        state = (torch.zeros_like(h_0), h_0)
        return annotations, mask, projected, state

    def step(self, word, state, annotations, mask, projected):
        """One decoder step: score, average, then advance.

        The order is the paper's and it is the part to read slowly. The score
        is taken against `state`, which at this point is s_{i-1}: the context
        for step i is chosen before step i runs. Luong et al. reverse it.
        """
        alpha = self.attention.weights(state[1], projected, mask)
        context = self.attention.context(alpha, annotations)
        merged = torch.cat([word, context], dim=-1)
        return self.decoder.step(merged, state), alpha

    def decode_forced(self, encoded, target_in: torch.Tensor) -> torch.Tensor:
        annotations, mask, projected, state = encoded
        embedded = self.target_embedding(target_in)
        outputs = []
        for t in range(target_in.shape[0]):
            state, _ = self.step(embedded[t], state, annotations, mask, projected)
            outputs.append(state[1])
        return self.readout(torch.stack(outputs))

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.decode_forced(self.encode(source), target[:-1])

    def loss(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = self(source, target)
        return nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target[1:].reshape(-1),
            ignore_index=PAD_INDEX,
        )


def greedy_decode(
    model: AttentionSeq2Seq,
    source: torch.Tensor,
    max_steps: int = 40,
    keep_weights: bool = False,
) -> tuple[list[list[int]], list[torch.Tensor]]:
    """Greedy decoding, optionally keeping every alpha vector on the way.

    The alignment matrix chapter 6 prints is exactly the stack of those
    vectors: row i is which source positions the model read while writing
    target word i.
    """
    sos = model.target_vocab.index["<sos>"]
    eos = model.target_vocab.index["<eos>"]
    batch = source.shape[1]

    with torch.no_grad():
        annotations, mask, projected, state = model.encode(source)
        token = torch.full((batch,), sos, dtype=torch.long)
        done = torch.zeros(batch, dtype=torch.bool)
        rows: list[list[int]] = [[] for _ in range(batch)]
        weights: list[torch.Tensor] = []
        for _ in range(max_steps):
            state, alpha = model.step(
                model.target_embedding(token), state, annotations, mask, projected
            )
            if keep_weights:
                weights.append(alpha)
            token = model.readout(state[1]).argmax(dim=-1)
            for column in range(batch):
                if done[column]:
                    continue
                if int(token[column]) == eos:
                    done[column] = True
                else:
                    rows[column].append(int(token[column]))
            if bool(done.all()):
                break
    return rows, weights


def greedy_accuracy(
    model: AttentionSeq2Seq,
    pairs,
    source_vocab: Vocabulary,
    target_vocab: Vocabulary,
    reverse_source: bool = False,
    batch_size: int = 64,
) -> tuple[float, list[tuple[int, bool]]]:
    """Exact match over `pairs`, plus (source length, hit) per sentence.

    Same scoring as chapter 5's, so the two chapters' tables are comparable
    figure for figure. See `seq2seq.greedy_accuracy` for why exact match rather
    than BLEU.
    """
    from .toy_corpus import _pad

    hits: list[tuple[int, bool]] = []
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        encoded = []
        for pair in chunk:
            tokens = list(pair.source)
            if reverse_source:
                tokens.reverse()
            encoded.append(source_vocab.encode(tokens))
        outputs, _ = greedy_decode(model, _pad(encoded))
        for pair, output in zip(chunk, outputs):
            hits.append((len(pair.source), output == target_vocab.encode(pair.target)))
    return sum(hit for _, hit in hits) / len(hits), hits


def train_one(
    seed: int,
    reverse_source: bool = False,
    n_hidden: int | None = None,
    epochs: int | None = None,
    train_pairs=None,
    bidirectional: bool = True,
):
    """One attention model under chapter 5's shared recipe, deterministic in `seed`.

    Everything that could differ between the two chapters' tables is taken from
    `seq2seq` rather than restated here: the split sizes, the batch size, the
    epoch count, the learning rate and the default width. A table comparing
    chapter 5's model with this one is otherwise comparing two recipes.
    """
    from .seq2seq import BATCH, EPOCHS, LEARNING_RATE, N_HIDDEN, N_TEST, N_TRAIN, train
    from .toy_corpus import batches, disjoint_splits, vocabularies

    source_vocab, target_vocab = vocabularies()
    if train_pairs is None:
        train_pairs, _ = disjoint_splits(N_TRAIN, N_TEST, seed=5)

    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(100 + seed)
    batched = batches(
        train_pairs, source_vocab, target_vocab, BATCH, generator,
        reverse_source=reverse_source,
    )
    model = AttentionSeq2Seq(
        source_vocab, target_vocab, n_hidden=n_hidden or N_HIDDEN,
        bidirectional=bidirectional,
    )
    losses = train(
        model, batched, epochs=epochs or EPOCHS, learning_rate=LEARNING_RATE
    )
    return model, losses
