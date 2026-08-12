"""Parameter counts and compute budgets, for chapter 9.

Chapter 8 turned table 1 of Vaswani et al. from asymptotics into counted
FLOPs. This module does the same job one level up, for the quantities the
pretraining literature is written in: how many parameters a published
configuration actually holds, and how much arithmetic training it on D tokens
actually costs.

Everything here is closed form. Nothing is trained and nothing is timed, so
every number this module produces is the same on every machine, to the last
digit. `experiments/ch09_counts.py` checks the FLOP side against
`torch.utils.flop_counter.FlopCounterMode` on a real stack, and
`tests/test_ch09.py` checks the parameter side against `sum(p.numel())` on
modules this repo builds - which is the only way to know a formula about
parameters is a formula about these parameters rather than about a
plausible-looking Transformer.

**Why the split between embedding and non-embedding matters.** Kaplan et al.
define N as the non-embedding parameter count and are explicit that the laws
do not hold for the other count. That is not bookkeeping: at GPT-1's size the
token embedding table is a quarter of the model, and at GPT-3's it is a third
of one percent. A rule fitted against one definition and applied to the other
is being applied across a factor that moves by two orders of magnitude over
the range it was fitted on. Every function here returns the split.

**The counting convention is chapter 8's**, unchanged: one multiply-add is 2
FLOPs, so an (m x k) by (k x p) product costs 2*m*k*p. Halve everything for
the multiply-accumulate convention.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParamCount:
    """A model's parameters, split the way the scaling literature splits them.

    `embedding` holds everything indexed by a vocabulary or a position:
    token embeddings, learned positional embeddings, segment embeddings. It
    is the part Kaplan et al. exclude from N.

    `blocks` is the repeated stack. `head` is whatever sits after it - a
    pooler, a masked-LM head, a final layer norm - and is separate because
    published counts disagree about whether to include it, which is a real
    part of why a rebuilt count misses a printed one.
    """

    embedding: int
    blocks: int
    head: int

    @property
    def non_embedding(self) -> int:
        """Kaplan's N."""
        return self.blocks + self.head

    @property
    def total(self) -> int:
        return self.embedding + self.blocks + self.head

    @property
    def embedding_fraction(self) -> float:
        return self.embedding / self.total


def block_params(
    d_model: int,
    d_ff: int | None = None,
    attention_bias: bool = True,
    ffn_bias: bool = True,
) -> int:
    """One pre-norm or post-norm block: attention, feed-forward, two norms.

    Four d x d projections, two feed-forward matrices of d x d_ff and
    d_ff x d, and two layer norms of 2d each. With biases everywhere and
    d_ff = 4*d this is 12*d^2 + 13*d, and the leading term is the 12*d^2 that
    makes Kaplan's 2N-per-token forward count come out.

    **The two bias switches are separate because this repo's own layer needs
    them separate.** `MultiHeadAttention` sets `bias=False` on all four
    projections, the way section 3.2.2's equations write them;
    `PositionwiseFeedForward` uses plain `nn.Linear` and carries both. So this
    repo is `(False, True)` and BERT, which carries biases throughout, is
    `(True, True)`. Writing it as one flag looks tidier and is wrong, which
    `tests/test_ch09.py` established the direct way: the single-flag version
    missed a built stack by exactly `n_layers * (d_ff + d_model)`.

    The whole question is worth 4*d or 5*d + d_ff per block, which at
    BERT-base's width is under a tenth of a percent - invisible at any
    rounding a paper prints, and the reason a rebuilt count can agree with a
    published one without either side having said what it counted.
    """
    if d_model <= 0:
        raise ValueError(f"need d_model > 0, got {d_model}")
    d_ff = 4 * d_model if d_ff is None else d_ff
    a = 1 if attention_bias else 0
    f = 1 if ffn_bias else 0
    attention = 4 * (d_model * d_model + a * d_model)
    feed_forward = (d_model * d_ff + f * d_ff) + (d_ff * d_model + f * d_model)
    norms = 2 * 2 * d_model
    return attention + feed_forward + norms


def decoder_only_params(
    n_layers: int,
    d_model: int,
    vocab: int,
    n_ctx: int,
    d_ff: int | None = None,
    attention_bias: bool = True,
    ffn_bias: bool = True,
    final_norm: bool = True,
) -> ParamCount:
    """A GPT-shaped stack: embeddings, N blocks, a final norm.

    The readout matrix is not counted, because every model in this family
    ties it to the token embedding table rather than allocating a second copy
    - which is also why the readout is free in parameters and not free in
    FLOPs, a gap `flops_per_token` keeps.
    """
    d_ff = 4 * d_model if d_ff is None else d_ff
    return ParamCount(
        embedding=vocab * d_model + n_ctx * d_model,
        blocks=n_layers * block_params(d_model, d_ff, attention_bias, ffn_bias),
        head=2 * d_model if final_norm else 0,
    )


def encoder_only_params(
    n_layers: int,
    d_model: int,
    vocab: int,
    n_ctx: int,
    d_ff: int | None = None,
    n_segments: int = 2,
    attention_bias: bool = True,
    ffn_bias: bool = True,
    pooler: bool = True,
    mlm_head: bool = False,
) -> ParamCount:
    """A BERT-shaped stack.

    Three embedding tables rather than two, because this family adds a
    segment embedding for the sentence-pair objective, and one layer norm on
    their sum.

    `pooler` and `mlm_head` are separate switches on purpose. A published
    parameter count for this family is ambiguous about both: the pooler is
    used by the sentence-level objective and by nothing else, and the
    masked-LM head is discarded before fine-tuning. Chapter 9 prints the
    count under each reading rather than picking one and calling the
    remainder a discrepancy.
    """
    d_ff = 4 * d_model if d_ff is None else d_ff
    b = 1 if ffn_bias else 0
    head = 0
    if pooler:
        head += d_model * d_model + b * d_model
    if mlm_head:
        # A dense d x d, a layer norm, and one output bias per vocabulary
        # entry. The output matrix itself is tied to the token embeddings.
        head += (d_model * d_model + b * d_model) + 2 * d_model + vocab
    return ParamCount(
        embedding=(
            vocab * d_model
            + n_ctx * d_model
            + n_segments * d_model
            + 2 * d_model  # layer norm on the summed embeddings
        ),
        blocks=n_layers * block_params(d_model, d_ff, attention_bias, ffn_bias),
        head=head,
    )


@dataclass(frozen=True)
class TokenFlops:
    """Forward matrix-multiply FLOPs for one token through a whole stack."""

    projections: int
    feed_forward: int
    attention: int

    @property
    def parametric(self) -> int:
        """The part proportional to the parameter count."""
        return self.projections + self.feed_forward

    @property
    def total(self) -> int:
        return self.parametric + self.attention

    @property
    def attention_share(self) -> float:
        return self.attention / self.total


def flops_per_token(
    n_layers: int, d_model: int, n_ctx: int, d_ff: int | None = None
) -> TokenFlops:
    """Forward FLOPs one token costs, summed over the stack.

    Per layer, per token: `8*d^2` for the four projections, `4*d*d_ff` for
    the feed-forward network, and `4*n*d` for the two products that carry the
    score matrix. The last one is the whole of chapter 8's quadratic term,
    divided by n.

    Read as a per-token cost it says something chapter 8 said per layer: with
    `d_ff = 4*d` the first two are `24*d^2` and the third is `4*n*d`, so the
    attention share is `n / (6*d + n)` and depends only on the ratio n/d.
    Half at `n = 6*d`, which is `2*d + d_ff` - the same crossover
    `cost.quadratic_half_point` returns, arrived at from the other side.
    """
    if n_layers <= 0 or d_model <= 0 or n_ctx < 0:
        raise ValueError(
            f"need n_layers > 0, d_model > 0, n_ctx >= 0, got "
            f"{n_layers}, {d_model}, {n_ctx}"
        )
    d_ff = 4 * d_model if d_ff is None else d_ff
    return TokenFlops(
        projections=n_layers * 8 * d_model * d_model,
        feed_forward=n_layers * 4 * d_model * d_ff,
        attention=n_layers * 4 * n_ctx * d_model,
    )


def attention_share(n_ctx: int, d_model: int, d_ff: int | None = None) -> float:
    """The fraction of a forward pass spent on the score matrix products.

    Closed form, `4*n*d / (8*d^2 + 4*d*d_ff + 4*n*d)`. At `d_ff = 4*d` this
    reduces to `n / (6*d + n)`. Independent of the number of layers, because
    every term scales with it.
    """
    d_ff = 4 * d_model if d_ff is None else d_ff
    f = flops_per_token(1, d_model, n_ctx, d_ff)
    return f.attention_share


def kaplan_compute(n_params: int, n_tokens: float) -> float:
    """`C = 6*N*D`, the approximation the scaling-law papers are written in.

    Two FLOPs per parameter per token forward, four backward. `n_params` is
    Kaplan's N, so pass `ParamCount.non_embedding` and not `.total`.

    What it drops is the attention term, which is not a function of the
    parameter count at all. `six_nd_error` says by how much.
    """
    if n_params <= 0 or n_tokens < 0:
        raise ValueError(f"need n_params > 0 and n_tokens >= 0")
    return 6.0 * n_params * n_tokens


def six_nd_error(
    n_layers: int,
    d_model: int,
    n_ctx: int,
    n_params: int,
    d_ff: int | None = None,
) -> float:
    """How much larger the exact count is than `6*N*D`, as a ratio.

    D cancels, and so does the factor of three between a forward pass and a
    forward-plus-backward pass, so this is just the exact per-token forward
    count divided by `2*N`. A ratio of 1.03 means 6ND is three percent low.

    The ratio is close to `1 + n/(6*d)` and is not exactly that, because N
    carries the biases and layer norms that the leading `12*d^2` per block
    does not.
    """
    exact = flops_per_token(n_layers, d_model, n_ctx, d_ff).total
    return exact / (2.0 * n_params)


# ---------------------------------------------------------------------------
# The two papers' fitted laws, as functions.
#
# Every constant below is transcribed from the paper named beside it, to the
# precision the paper prints and no further. Both papers write their
# single-variable exponents with "~" rather than "=", so a third significant
# figure here would be invented rather than measured.
# ---------------------------------------------------------------------------

#: Kaplan et al. (2020), equation 1.1. N is the *non-embedding* parameter
#: count; the paper is explicit that the law does not hold for the other one.
KAPLAN_ALPHA_N = 0.076
KAPLAN_N_C = 8.8e13

#: Equation 1.2. D in tokens.
KAPLAN_ALPHA_D = 0.095
KAPLAN_D_C = 5.4e13

#: Equation 1.3, and note this one attaches to C_min rather than to C. The
#: paper fits a law for plain C too (appendix A, table 5: alpha 0.057,
#: C_c 1.6e7) and says in footnote 3 that it is the C_min trend "that should
#: be used to make predictions". Units are PF-days.
KAPLAN_ALPHA_C = 0.050
KAPLAN_C_C = 3.1e8

#: Appendix A, table 6: the compute-optimal allocation. C_min in PF-days,
#: N_opt in parameters, D_opt in tokens.
KAPLAN_P_N, KAPLAN_N_E = 0.73, 1.3e9
KAPLAN_P_D, KAPLAN_D_E = 0.27, 2.0e10

#: Hoffmann et al. (2022), appendix D.2 equation 10. N in parameters and D in
#: tokens, both counted *including* embeddings - the opposite convention to
#: Kaplan's, which appendix F states and which neither paper connects to the
#: exponents they disagree about.
CHINCHILLA_E = 1.69
CHINCHILLA_A = 406.4
CHINCHILLA_ALPHA = 0.34
CHINCHILLA_B = 410.7
CHINCHILLA_BETA = 0.28


def kaplan_loss_from_params(n_params: float) -> float:
    """`L(N)` of equation 1.1, in nats per token."""
    return (KAPLAN_N_C / n_params) ** KAPLAN_ALPHA_N


def kaplan_loss_from_tokens(n_tokens: float) -> float:
    """`L(D)` of equation 1.2."""
    return (KAPLAN_D_C / n_tokens) ** KAPLAN_ALPHA_D


def kaplan_loss_from_compute(pf_days: float) -> float:
    """`L(C_min)` of equation 1.3. Argument in petaflop-days."""
    return (KAPLAN_C_C / pf_days) ** KAPLAN_ALPHA_C


def kaplan_optimal(pf_days: float) -> tuple[float, float]:
    """Kaplan's compute-optimal `(N, D)` at a budget, from table 6."""
    return (
        KAPLAN_N_E * pf_days**KAPLAN_P_N,
        KAPLAN_D_E * pf_days**KAPLAN_P_D,
    )


def chinchilla_loss(n_params: float, n_tokens: float) -> float:
    """`L(N, D) = E + A/N^alpha + B/D^beta`, appendix D.2 equation 10.

    The irreducible term E is 1.69 nats, which is what the fit says no model
    of any size trained on any amount of this data can go below. Both other
    terms are positive, so the function is monotone decreasing in each
    argument - it can rank two configurations but it cannot say either is
    good.
    """
    if n_params <= 0 or n_tokens <= 0:
        raise ValueError("need n_params > 0 and n_tokens > 0")
    return (
        CHINCHILLA_E
        + CHINCHILLA_A / n_params**CHINCHILLA_ALPHA
        + CHINCHILLA_B / n_tokens**CHINCHILLA_BETA
    )


def chinchilla_frontier_exponents() -> tuple[float, float]:
    """`(a, b)` with `N_opt ~ C^a` and `D_opt ~ C^b`, from alpha and beta.

    Equation 4 of the paper: `a = beta / (alpha + beta)` and
    `b = alpha / (alpha + beta)`, so the two sum to exactly 1 by
    construction - which is the whole claim, since it means the two factors
    of a compute budget are split between model and data and nothing else.

    Deriving them rather than hard-coding table 2's rounded 0.46 and 0.54 is
    the point: it checks that the paper's own two halves agree.
    """
    total = CHINCHILLA_ALPHA + CHINCHILLA_BETA
    return CHINCHILLA_BETA / total, CHINCHILLA_ALPHA / total


def chinchilla_optimal(n_params: float, tokens_per_param: float = 20.0):
    """The rule of thumb, stated as what it is: a reading of table 3.

    The paper never writes "twenty tokens per parameter" anywhere. Its table
    3 projects nine compute budgets and the ratio implied by those rows runs
    from 20.0 at 400M to 21.6 at 10T, drifting upward with scale rather than
    holding. `experiments/ch09_laws.py` prints the nine ratios so the reader
    can see the spread the rule of thumb flattens.
    """
    return n_params * tokens_per_param
