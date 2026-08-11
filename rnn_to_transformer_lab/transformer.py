"""The Transformer of Vaswani et al. (2017), built from the paper's equations.

Chapter 6 ends on an observation about its own equation (5): the weighted sum

    c_i = sum_j alpha_ij h_j

has no `t-1` anywhere in it. The only reason it has to wait is that the h_j
come out of a recurrence one at a time. Take the recurrence away and let every
position attend to every position of its *own* sequence, and nothing waits.
That is self-attention, and this module is what the paper builds out of it.

Everything here follows the paper. The places it does not are listed at the
bottom of this docstring, each with the reason, because a reader comparing this
code with the PDF should not have to discover a difference by being confused.

**Scaled dot-product attention, section 3.2.1, equation (1).**

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

`scaled_dot_product` below is that line. The division is the part chapter 7
derives rather than states: if the components of q and k are independent with
mean 0 and variance 1, then q . k has mean 0 and variance d_k, so the scores
grow like sqrt(d_k) and the softmax saturates. The paper puts that argument in
a footnote; the chapter measures it.

**Multi-head attention, section 3.2.2.** h heads, each of width d_k = d_v =
d_model / h, run in parallel and concatenated:

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    head_i             = Attention(Q W_i^Q, K W_i^K, V W_i^V)

`MultiHeadAttention` fuses the h projections into one Linear of width d_model
and reshapes, which is arithmetically the h separate projections of the paper
stacked side by side. The fused form is also what makes the paper's own
accounting visible: because d_k shrinks as h grows, the parameter count does
not depend on h at all.

**The projections carry no bias, and the feed-forward network does.** The paper
writes the attention projections as bare matrices W_i^Q, W_i^K, W_i^V, W^O and
writes the feed-forward network with its biases spelled out, equation (2):

    FFN(x) = max(0, x W_1 + b_1) W_2 + b_2

so that is what is here. It is a small thing that changes a parameter count,
and the chapter prints parameter counts.

**Post-LN, section 3.1.** The output of each sub-layer is

    LayerNorm(x + Sublayer(x))

with the normalization *after* the residual addition. Later work moved it
before, and this repo keeps the paper's order; chapter 8 is where that
difference is measured rather than asserted.

**Sinusoidal positional encoding, section 3.5.**

    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

Self-attention has no notion of order: permute the input and the output
permutes with it, exactly. `positional_encoding` is what breaks that symmetry,
and `tests/test_ch07.py` asserts both halves - that the model without it is
permutation-equivariant, and that with it, it is not.

**Masking, section 3.2.3.** The decoder's self-attention masks out
"all values in the input of the softmax which correspond to illegal
connections", which is inside the softmax rather than after it. Chapter 6 hit
the same seam with padding and the failure mode is the same: nothing raises,
the loss looks fine, and a few points of accuracy are gone.

Departures from the paper, all of them forced by this repo's constraints:

* **Size.** The paper is d_model 512, 6 layers each side, 8 heads, d_ff 2048,
  and 12 hours on eight GPUs. This runs a few dozen units on a laptop CPU in
  well under a minute, because the book promised a reader with no graphics
  card that every experiment finishes in minutes.
* **No shared embedding weights.** Section 3.4 ties the two embedding matrices
  and the pre-softmax projection together. This corpus has an English source
  vocabulary and a Vietnamese target vocabulary with no token in common, so
  there is nothing to tie. The sqrt(d_model) scaling from the same section is
  kept.
* **Dropout defaults to zero.** Section 5.4 applies P_drop = 0.1 to every
  sub-layer output and to the embedding sums. The mechanism is implemented and
  wired to the paper's three places; the default is off because this corpus is
  a finite grammar seen thousands of times, where dropout only removes signal.
  Chapter 8 is where the regularization is turned on and measured.
* **The optimizer is this repo's shared recipe, not the paper's.** Section 5.3
  uses Adam with a warmup schedule over 4000 steps. `train_one` takes the
  learning rate, the batch size, the epoch count and the split from `seq2seq`,
  unchanged, for the reason chapter 6 gives: two chapters whose tables sit side
  by side have to differ in one thing only, and here that thing is the
  architecture. The warmup schedule is chapter 8's subject.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .seq2seq import PAD_INDEX
from .toy_corpus import Vocabulary

#: Section 3.5's constant, the base of the geometric progression of
#: wavelengths. Named rather than inlined because chapter 7 works out what
#: wavelength range it actually produces, and an exercise changes it.
PE_BASE = 10000.0


def scaled_dot_product(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Equation (1). Trailing dimension is d_k; `mask` is True where allowed.

    The mask is applied to the scores, before the softmax, which is what
    section 3.2.3 says: masked positions are set to -inf "in the input of the
    softmax". Masking the weights afterwards and renormalising gives a
    different answer whenever the masked entries were not already negligible,
    and it is a change nothing in a training run reports.
    """
    d_k = query.shape[-1]
    scores = query @ key.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ value, weights


def positional_encoding(n_positions: int, d_model: int) -> torch.Tensor:
    """Section 3.5's table, (n_positions, d_model).

    Column 2i and column 2i+1 share one frequency, sine in the even column and
    cosine in the odd one, so the table is d_model/2 rotating pairs rather than
    d_model independent curves. That pairing is the whole of the linear-shift
    property chapter 7 checks: shifting the position by k turns each pair by a
    fixed angle k * omega_i, and a rotation is a linear map that does not
    depend on where it started.
    """
    if d_model % 2:
        raise ValueError(f"d_model must be even for the sin/cos pairs, got {d_model}")
    position = torch.arange(n_positions, dtype=torch.float32).unsqueeze(1)
    pair = torch.arange(0, d_model, 2, dtype=torch.float32)
    angle = position / torch.pow(PE_BASE, pair / d_model)
    table = torch.zeros(n_positions, d_model)
    table[:, 0::2] = torch.sin(angle)
    table[:, 1::2] = torch.cos(angle)
    return table


def causal_mask(n_steps: int) -> torch.Tensor:
    """(1, 1, n_steps, n_steps), True on and below the diagonal.

    Query position i may read key positions 0 to i inclusive. Reading its own
    position is allowed and reading position i+1 is not, which combined with
    the target being offset by one is what section 3.2.3 means by the
    prediction for position i depending only on outputs before i.
    """
    square = torch.ones(n_steps, n_steps, dtype=torch.bool)
    return torch.tril(square).view(1, 1, n_steps, n_steps)


def padding_mask(tokens: torch.Tensor) -> torch.Tensor:
    """(batch, 1, 1, steps) from a (batch, steps) index tensor.

    Broadcast over heads and over query positions: every query in the batch row
    is forbidden the same padded keys.
    """
    return (tokens != PAD_INDEX).view(tokens.shape[0], 1, 1, tokens.shape[1])


class MultiHeadAttention(nn.Module):
    """Section 3.2.2. h heads of width d_model / h, concatenated and projected.

    The four projections are stored fused, one Linear of width d_model each,
    and split into heads by a reshape. That is the same computation as the
    paper's h separate matrices: head i reads rows i*d_k to (i+1)*d_k of the
    fused weight, and no head ever sees another head's rows.

    No biases, because the paper writes these four as matrices and writes the
    feed-forward network's biases out explicitly two equations later.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} is not divisible by n_heads {n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, steps, d_model) -> (batch, heads, steps, d_k)."""
        batch, steps, _ = x.shape
        return x.view(batch, steps, self.n_heads, self.d_k).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch-first in and out; `mask` broadcasts to (batch, heads, q, k).

        Returns the projected output and the per-head attention weights, the
        second only so that chapter 7 can print what a head reads without
        instrumenting the model.
        """
        batch, steps, _ = query.shape
        heads, weights = scaled_dot_product(
            self._split(self.w_q(query)),
            self._split(self.w_k(key)),
            self._split(self.w_v(value)),
            mask,
        )
        joined = heads.transpose(1, 2).reshape(batch, steps, self.d_model)
        return self.dropout(self.w_o(joined)), weights


class PositionwiseFeedForward(nn.Module):
    """Equation (2): max(0, x W_1 + b_1) W_2 + b_2, applied at each position.

    "Position-wise" is the whole content of the name. It is one two-layer
    network with one set of weights, run at every position separately, so it
    moves nothing between positions: in this architecture the only thing that
    moves information sideways is attention. The paper also describes it as
    two convolutions with kernel size 1, which is the same statement.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.inner = nn.Linear(d_model, d_ff)
        self.outer = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.outer(torch.relu(self.inner(x))))


class EncoderLayer(nn.Module):
    """Section 3.1: self-attention, then the feed-forward network, both post-LN."""

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm_attention = nn.LayerNorm(d_model)
        self.norm_feed_forward = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attended, weights = self.self_attention(x, x, x, mask)
        x = self.norm_attention(x + attended)
        x = self.norm_feed_forward(x + self.feed_forward(x))
        return x, weights


class DecoderLayer(nn.Module):
    """Three sub-layers: masked self-attention, cross-attention, feed-forward.

    The middle one is where the source enters, and it is the only sub-layer in
    the decoder that reads anything the decoder did not produce: queries come
    from the layer below, keys and values from the encoder's output. That is
    the same shape as chapter 6's mechanism, and section 3.2.3 says so - "this
    mimics the typical encoder-decoder attention mechanisms in sequence-to-
    sequence models". What is new is the sub-layer either side of it.
    """

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm_self = nn.LayerNorm(d_model)
        self.norm_cross = nn.LayerNorm(d_model)
        self.norm_feed_forward = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        target_mask: torch.Tensor,
        source_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attended, _ = self.self_attention(x, x, x, target_mask)
        x = self.norm_self(x + attended)
        crossed, cross_weights = self.cross_attention(x, memory, memory, source_mask)
        x = self.norm_cross(x + crossed)
        x = self.norm_feed_forward(x + self.feed_forward(x))
        return x, cross_weights


class Transformer(nn.Module):
    """The encoder-decoder of figure 1, at a size that fits the book's budget.

    The boundary keeps this repo's (steps, batch) convention so that
    `toy_corpus.batches` and chapter 6's scoring functions work unchanged;
    everything inside is batch-first, because attention is unreadable with the
    batch in the middle.
    """

    def __init__(
        self,
        source_vocab: Vocabulary,
        target_vocab: Vocabulary,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int | None = None,
        dropout: float = 0.0,
        max_positions: int = 64,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        # Section 3.3 sets d_ff to four times d_model (2048 against 512), so
        # the ratio is the paper's even though the sizes are not.
        self.d_ff = d_ff if d_ff is not None else 4 * d_model
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab

        self.source_embedding = nn.Embedding(
            len(source_vocab), d_model, padding_idx=PAD_INDEX
        )
        self.target_embedding = nn.Embedding(
            len(target_vocab), d_model, padding_idx=PAD_INDEX
        )
        self.register_buffer(
            "positions", positional_encoding(max_positions, d_model), persistent=False
        )
        self.embedding_dropout = nn.Dropout(dropout)

        self.encoder_layers = nn.ModuleList(
            EncoderLayer(d_model, n_heads, self.d_ff, dropout)
            for _ in range(n_layers)
        )
        self.decoder_layers = nn.ModuleList(
            DecoderLayer(d_model, n_heads, self.d_ff, dropout)
            for _ in range(n_layers)
        )
        self.readout = nn.Linear(d_model, len(target_vocab))

    def _embed(self, tokens: torch.Tensor, embedding: nn.Embedding) -> torch.Tensor:
        """Section 3.4's sqrt(d_model) scaling, then the positional table.

        The scaling matters more than it looks. An embedding row initialised at
        unit scale and a positional entry in [-1, 1] are the same size, so
        without the multiplication the position would carry as much of the
        vector as the word does.
        """
        steps = tokens.shape[1]
        scaled = embedding(tokens) * math.sqrt(self.d_model)
        return self.embedding_dropout(scaled + self.positions[:steps])

    def encode(self, source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(steps, batch) indices in; memory (batch, steps, d_model) and its mask."""
        tokens = source.transpose(0, 1)
        mask = padding_mask(tokens)
        x = self._embed(tokens, self.source_embedding)
        for layer in self.encoder_layers:
            x, _ = layer(x, mask)
        return x, mask

    def decode(
        self,
        memory: torch.Tensor,
        source_mask: torch.Tensor,
        target_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Logits (steps, batch, vocab) and the last layer's cross-attention."""
        tokens = target_in.transpose(0, 1)
        target_mask = causal_mask(tokens.shape[1]).to(memory.device)
        x = self._embed(tokens, self.target_embedding)
        weights = None
        for layer in self.decoder_layers:
            x, weights = layer(x, memory, target_mask, source_mask)
        return self.readout(x).transpose(0, 1), weights

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        memory, source_mask = self.encode(source)
        logits, _ = self.decode(memory, source_mask, target[:-1])
        return logits

    def loss(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = self(source, target)
        return nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target[1:].reshape(-1),
            ignore_index=PAD_INDEX,
        )


def greedy_decode(
    model: Transformer,
    source: torch.Tensor,
    max_steps: int = 40,
    keep_weights: bool = False,
) -> tuple[list[list[int]], list[torch.Tensor]]:
    """Greedy decoding, same interface as chapter 6's so the tables compare.

    One difference from every earlier model in this repo, and it is worth
    seeing rather than hiding: there is no state to carry forward. Each step
    re-runs the decoder over the entire prefix, so decoding step i costs
    O(i) work and the whole sentence costs O(T^2) even though a recurrent
    decoder would have cost O(T). Training is where the Transformer wins the
    time back, because there every step runs at once. What removes the waste at
    decoding time is the key-value cache, which is chapter 15's subject.
    """
    sos = model.target_vocab.index["<sos>"]
    eos = model.target_vocab.index["<eos>"]
    batch = source.shape[1]

    with torch.no_grad():
        memory, source_mask = model.encode(source)
        prefix = torch.full((1, batch), sos, dtype=torch.long)
        done = torch.zeros(batch, dtype=torch.bool)
        rows: list[list[int]] = [[] for _ in range(batch)]
        weights: list[torch.Tensor] = []
        for _ in range(max_steps):
            logits, cross = model.decode(memory, source_mask, prefix)
            if keep_weights:
                weights.append(cross[:, :, -1, :])
            token = logits[-1].argmax(dim=-1)
            for column in range(batch):
                if done[column]:
                    continue
                if int(token[column]) == eos:
                    done[column] = True
                else:
                    rows[column].append(int(token[column]))
            if bool(done.all()):
                break
            prefix = torch.cat([prefix, token.unsqueeze(0)], dim=0)
    return rows, weights


def greedy_accuracy(
    model: Transformer,
    pairs,
    source_vocab: Vocabulary,
    target_vocab: Vocabulary,
    reverse_source: bool = False,
    batch_size: int = 64,
) -> tuple[float, list[tuple[int, bool]]]:
    """Exact match over `pairs`, plus (source length, hit) per sentence.

    Byte for byte the scoring chapters 5 and 6 use, so a row of chapter 7's
    table means the same thing as a row of theirs.
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
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    d_ff: int | None = None,
    epochs: int | None = None,
    train_pairs=None,
    dropout: float = 0.0,
):
    """One Transformer under chapters 5 and 6's shared recipe, given `seed`.

    Split sizes, batch size, epoch count, learning rate and the optimizer all
    come from `seq2seq` unchanged, for the reason chapter 6's `train_one` gives
    and for one more: the paper's own schedule is chapter 8's subject, and a
    chapter 7 table trained under it could not be set beside chapter 6's.
    """
    from .seq2seq import BATCH, EPOCHS, LEARNING_RATE, N_TEST, N_TRAIN, train
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
    model = Transformer(
        source_vocab, target_vocab, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, d_ff=d_ff, dropout=dropout,
    )
    losses = train(
        model, batched, epochs=epochs or EPOCHS, learning_rate=LEARNING_RATE
    )
    return model, losses
