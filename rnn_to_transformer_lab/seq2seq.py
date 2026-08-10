"""The encoder-decoder of Sutskever, Vinyals and Le (2014), at toy scale.

One LSTM reads the source sentence and its final state *is* the whole meaning
of that sentence; a second LSTM starts from that state and writes the target.
Nothing else passes between them. The paper puts it in one line: the model
computes the fixed-dimensional representation v of the input, "given by the
last hidden state of the LSTM", and then runs a language model whose initial
hidden state is set to v.

Three things here follow the paper deliberately, and one departs from it.

**Two separate LSTMs, not one.** Section 2, difference one: "we used two
different LSTMs: one for the input sequence and another for the output
sequence, because doing so increases the number model parameters at negligible
computational cost". `Seq2Seq` therefore carries `encoder` and `decoder` as
independent modules with independent weights.

**The context arrives once, as the initial state.** It is not fed again at each
decoder step. That is what makes the vector a bottleneck rather than a hint,
and it is the difference between this model and Cho et al. (2014a), whose
decoder reads the context at every step. Chapter 6 removes the bottleneck; this
module is what it removes it from.

**The source may be reversed.** Section 3.3. Reversing is a property of the
data rather than of the model, so it lives in `toy_corpus.batches` and this
module never knows which it was handed.

**The departure: no peepholes, and one layer.** Sutskever et al. say they use
"the LSTM formulation from Graves", and Graves (2013) equations (7) to (11)
carry both a forget gate and peephole connections (the diagonal W_ci, W_cf and
W_co terms). This module implements the forget gate and not the peepholes, and
uses one layer where the paper uses four. Both are size decisions forced by the
book's CPU budget, not disagreements with the paper, and chapter 5 states them
in the prose rather than leaving the listing to imply the paper was simpler
than it is.

Style note. Chapter 4's `lstm.py` keeps its cell as a dataclass of loose
tensors, because that chapter reaches in and sets weights by hand as often as
it trains them. Chapter 5 only ever trains, so these are `nn.Module`s and the
four gate blocks are fused into one matrix multiply: at four separate matmuls
per step the Python overhead, not the arithmetic, is what the budget goes on.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .toy_corpus import Vocabulary

PAD_INDEX = 0


class LstmLayer(nn.Module):
    """The 2000 cell with tanh: c_t = f_t c_{t-1} + i_t tanh(a_c), h_t = o_t tanh(c_t).

    Chapter 4 derived the 1997 cell, whose self-connection is a fixed 1.0 and
    whose squashing functions are scaled logistics rather than tanh. This is
    the version the field converged on and the one `torch.nn.LSTM` implements:
    the forget gate of Gers, Schmidhuber and Cummins (2000), and tanh in both
    places. Chapter 4's bridge box is where the swap is argued; this class is
    where the book starts using it.

    The four blocks are stacked into one weight matrix in the order
    (input, forget, cell, output), which is also the order `torch.nn.LSTM`
    stacks them in, so a reader comparing the two is comparing like with like.
    """

    def __init__(self, n_input: int, n_hidden: int) -> None:
        super().__init__()
        self.n_hidden = n_hidden
        self.w_x = nn.Parameter(torch.empty(4 * n_hidden, n_input))
        self.w_h = nn.Parameter(torch.empty(4 * n_hidden, n_hidden))
        self.bias = nn.Parameter(torch.zeros(4 * n_hidden))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Uniform in +/- 1/sqrt(n_hidden), then the forget bias set to 1.

        The interval is what `torch.nn.LSTM` uses. The forget bias is not:
        starting it at 1 holds the gate open at the beginning of training, so
        the cell behaves like chapter 4's fixed 1.0 until gradients have a
        reason to close it. Without it a forget gate initialised at 0.5
        multiplies the state by a half every step and the carousel leaks from
        the first batch.
        """
        bound = 1.0 / math.sqrt(self.n_hidden)
        for weight in (self.w_x, self.w_h):
            nn.init.uniform_(weight, -bound, bound)
        nn.init.zeros_(self.bias)
        with torch.no_grad():
            self.bias[self.n_hidden : 2 * self.n_hidden].fill_(1.0)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """The part of the gate pre-activations that does not depend on h.

        Split out so a whole sequence can be projected in one matrix multiply
        before the recurrence starts. Only the `h_{t-1}` term is genuinely
        sequential, and at this size the Python loop rather than the arithmetic
        is what the budget goes on, so halving the number of matmuls inside the
        loop is worth the extra method.
        """
        return x @ self.w_x.T + self.bias

    def step_projected(
        self, projected: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        c_prev, h_prev = state
        gates = projected + h_prev @ self.w_h.T
        i, f, g, o = gates.chunk(4, dim=-1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        c = f * c_prev + i * torch.tanh(g)
        h = o * torch.tanh(c)
        return c, h

    def step(
        self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One step from a raw input. What a decoder generating token by token uses."""
        return self.step_projected(self.project(x), state)

    def initial_state(
        self, batch: int, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zero = torch.zeros(batch, self.n_hidden, dtype=dtype)
        return zero, zero


class Seq2Seq(nn.Module):
    """Encoder LSTM, decoder LSTM, and a fixed-width vector between them.

    `n_hidden` is the width of that vector, and it is the knob chapter 5 turns:
    the context the decoder receives is (c_T, h_T), so it is exactly
    `2 * n_hidden` real numbers however long the source sentence was. The
    paper's own model carries four layers of a thousand cells, which is where
    its "8000 real numbers to represent a sentence" comes from.
    """

    def __init__(
        self,
        source_vocab: Vocabulary,
        target_vocab: Vocabulary,
        n_hidden: int,
        n_embedding: int = 32,
    ) -> None:
        super().__init__()
        self.n_hidden = n_hidden
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.source_embedding = nn.Embedding(len(source_vocab), n_embedding, padding_idx=PAD_INDEX)
        self.target_embedding = nn.Embedding(len(target_vocab), n_embedding, padding_idx=PAD_INDEX)
        self.encoder = LstmLayer(n_embedding, n_hidden)
        self.decoder = LstmLayer(n_embedding, n_hidden)
        self.readout = nn.Linear(n_hidden, len(target_vocab))

    @property
    def context_width(self) -> int:
        """How many real numbers the decoder gets, whatever the sentence."""
        return 2 * self.n_hidden

    def encode(self, source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Read (steps, batch) of source indices; return the state at the last real token.

        The state is frozen wherever the input is padding, so a short sentence
        in a batch of long ones ends with the state it had at its own final
        word rather than one carried through several steps of `<pad>`. Getting
        this wrong is invisible in the loss and shows up only as an accuracy
        that depends on what a sentence was batched with.
        """
        projected = self.encoder.project(self.source_embedding(source))
        state = self.encoder.initial_state(source.shape[1], projected.dtype)
        for t in range(source.shape[0]):
            live = (source[t] != PAD_INDEX).unsqueeze(-1).to(projected.dtype)
            c_next, h_next = self.encoder.step_projected(projected[t], state)
            state = (
                live * c_next + (1.0 - live) * state[0],
                live * h_next + (1.0 - live) * state[1],
            )
        return state

    def decode_forced(
        self, context: tuple[torch.Tensor, torch.Tensor], target_in: torch.Tensor
    ) -> torch.Tensor:
        """Teacher forcing: (steps, batch) of target indices in, logits out."""
        projected = self.decoder.project(self.target_embedding(target_in))
        state = context
        outputs = []
        for t in range(target_in.shape[0]):
            state = self.decoder.step_projected(projected[t], state)
            outputs.append(state[1])
        return self.readout(torch.stack(outputs))

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Logits for `target[1:]`, given `target[:-1]` and the source."""
        return self.decode_forced(self.encode(source), target[:-1])

    def loss(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = self(source, target)
        return nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target[1:].reshape(-1),
            ignore_index=PAD_INDEX,
        )


def greedy_decode(
    model: Seq2Seq, source: torch.Tensor, max_steps: int = 40
) -> list[list[int]]:
    """Beam search with B = 1, written out separately because it is the baseline.

    Takes a padded (steps, batch) source and returns one index list per column,
    with <sos> and <eos> stripped.
    """
    sos = model.target_vocab.index["<sos>"]
    eos = model.target_vocab.index["<eos>"]
    batch = source.shape[1]

    with torch.no_grad():
        state = model.encode(source)
        token = torch.full((batch,), sos, dtype=torch.long)
        done = torch.zeros(batch, dtype=torch.bool)
        rows: list[list[int]] = [[] for _ in range(batch)]
        for _ in range(max_steps):
            state = model.decoder.step(model.target_embedding(token), state)
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
    return rows


def beam_decode(
    models: list[Seq2Seq],
    source: torch.Tensor,
    beam: int,
    max_steps: int = 40,
    length_normalize: bool = False,
) -> list[int]:
    """One sentence, `beam` hypotheses, one or more models averaged.

    This is the paper's decoder of section 3.2, written for a single sentence
    so that the mechanism is readable: keep B partial hypotheses, extend each
    one by every word in the vocabulary, keep the B best by log probability,
    and retire a hypothesis to the finished set as soon as it emits <eos>.

    With several models the log probabilities are averaged before the ranking,
    which is what an ensemble is: not a vote on the final sentence, but one
    distribution per step that every member helped shape. The paper ensembles
    five LSTMs "that differ in their random initializations and in the random
    order of minibatches".

    `source` is (steps, 1). `length_normalize` divides a finished hypothesis's
    score by its length; the paper does not do this and the default follows it,
    but chapter 5 measures what it costs.
    """
    sos = models[0].target_vocab.index["<sos>"]
    eos = models[0].target_vocab.index["<eos>"]

    with torch.no_grad():
        states = [model.encode(source) for model in models]
        # A live hypothesis is (tokens, score, per-model decoder state). The
        # state belongs to the prefix, so every extension of one hypothesis
        # shares it.
        live: list[tuple[list[int], float, list[tuple[torch.Tensor, torch.Tensor]]]] = [
            ([], 0.0, states)
        ]
        finished: list[tuple[list[int], float]] = []
        best_finished = -float("inf")

        for _ in range(max_steps):
            if not live:
                break
            candidates: list[tuple[list[int], float, list[tuple[torch.Tensor, torch.Tensor]]]] = []
            for tokens, score, hypothesis_states in live:
                previous = torch.tensor([tokens[-1] if tokens else sos], dtype=torch.long)
                next_states = []
                total = None
                for model, state in zip(models, hypothesis_states):
                    stepped = model.decoder.step(model.target_embedding(previous), state)
                    next_states.append(stepped)
                    logp = torch.log_softmax(model.readout(stepped[1]), dim=-1)
                    total = logp if total is None else total + logp
                logp = (total / len(models)).squeeze(0)

                # Only the best `beam` extensions of any one hypothesis can
                # reach the global top `beam`, so there is no need to score the
                # rest of the vocabulary against each other.
                top = torch.topk(logp, k=min(beam, logp.shape[0]))
                for value, index in zip(top.values.tolist(), top.indices.tolist()):
                    candidates.append((tokens + [index], score + value, next_states))

            candidates.sort(key=lambda item: item[1], reverse=True)
            live = []
            for tokens, score, hypothesis_states in candidates[:beam]:
                if tokens[-1] == eos:
                    finished.append((tokens[:-1], score))
                    best_finished = max(best_finished, score)
                else:
                    live.append((tokens, score, hypothesis_states))

            # A log probability only ever decreases, so a live hypothesis
            # already behind a finished one can never catch it. Dropping those
            # is exact rather than a heuristic, and it is what lets the search
            # stop early without the premature stop that costs accuracy: ending
            # the moment `beam` hypotheses have finished throws away live ones
            # that were still ahead of every one of them.
            live = [item for item in live if item[1] > best_finished]

    pool = finished if finished else [(tokens, score) for tokens, score, _ in live]
    if not pool:
        return []
    if length_normalize:
        return max(pool, key=lambda item: item[1] / max(len(item[0]), 1))[0]
    return max(pool, key=lambda item: item[1])[0]


def exact_match(model_output: list[int], reference: list[int]) -> bool:
    return model_output == reference


def greedy_accuracy(
    model: Seq2Seq, pairs, source_vocab: Vocabulary, target_vocab: Vocabulary,
    reverse_source: bool = False, batch_size: int = 64,
) -> tuple[float, list[tuple[int, bool]]]:
    """Exact-match accuracy over `pairs`, plus (source length, hit) per sentence.

    Exact match rather than BLEU, deliberately. BLEU on a corpus this small and
    this regular measures nothing a partial credit score should be trusted for,
    and the failure this chapter cares about is a model that puts the adjective
    on the wrong side, which is a wrong sentence rather than a slightly worse
    one.
    """
    from .toy_corpus import _pad  # local: the padding layout belongs to the corpus

    hits: list[tuple[int, bool]] = []
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        encoded = []
        for pair in chunk:
            tokens = list(pair.source)
            if reverse_source:
                tokens.reverse()
            encoded.append(source_vocab.encode(tokens))
        outputs = greedy_decode(model, _pad(encoded))
        for pair, output in zip(chunk, outputs):
            hits.append((len(pair.source), output == target_vocab.encode(pair.target)))
    return sum(hit for _, hit in hits) / len(hits), hits


#: The one configuration chapter 5's tables share.
#:
#: Four experiment scripts print numbers the chapter sets side by side, so they
#: have to be numbers from the same setup: a beam-search gain measured on a
#: model trained for twice as long as the one in the reversal table is not
#: comparable with it. Everything that could differ between those scripts is
#: pinned here instead of copied into each of them.
#:
#: Sized against the book's CPU budget rather than against convergence. Fourteen
#: epochs leaves the single model short of what it would reach given twenty, and
#: chapter 5 says so; what matters for every comparison in the chapter is that
#: both sides of it stop at the same place.
N_TRAIN = 6000
N_TEST = 300
BATCH = 128
EPOCHS = 14
LEARNING_RATE = 0.005
N_HIDDEN = 128


def train_one(
    seed: int,
    reverse_source: bool,
    n_hidden: int = N_HIDDEN,
    epochs: int = EPOCHS,
    train_pairs=None,
) -> tuple[Seq2Seq, list[float]]:
    """One model under the shared recipe, deterministic in `seed`.

    The seed moves both the weight initialisation and the order of the
    minibatches, which is the pair of things Sutskever et al. vary between the
    five members of their ensemble: models "that differ in their random
    initializations and in the random order of minibatches".
    """
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
    model = Seq2Seq(source_vocab, target_vocab, n_hidden=n_hidden)
    losses = train(model, batched, epochs=epochs, learning_rate=LEARNING_RATE)
    return model, losses


def train(
    model: Seq2Seq,
    batched,
    epochs: int,
    learning_rate: float = 0.01,
    clip: float = 5.0,
) -> list[float]:
    """Adam, gradient-norm clipping, no learning rate schedule.

    The clip is the paper's, section 3.4: "we enforced a hard constraint on the
    norm of the gradient by scaling it when its norm exceeded a threshold ...
    If s > 5, we set g = 5g/s". Chapter 3 derived why that works. The optimizer
    is not the paper's, which is plain SGD at a learning rate of 0.7 halved
    every half epoch after the fifth; Adam is here because the book's budget is
    seconds rather than ten days, and chapter 5 says so.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses: list[float] = []
    for _ in range(epochs):
        for source, target in batched:
            loss = model.loss(source, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            losses.append(loss.item())
    return losses
