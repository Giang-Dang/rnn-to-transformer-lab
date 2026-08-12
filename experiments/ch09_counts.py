"""Chapter 9: four papers' headline counts, rebuilt from their own tables.

Nothing here trains or times anything, so every number is reproducible on any
machine to the last digit.

1. Every published parameter count in the chapter's four papers, rebuilt from
   the hyperparameters those same papers print. The interesting output is not
   that they agree - it is the two that do not.
2. The embedding table as a fraction of the model, across three orders of
   magnitude, which is why Kaplan et al. define N without it.
3. C = 6ND against an exact forward count, and the closed form of the gap.
4. Kaplan's own table 1, rebuilt term by term, against the same exact count.
5. What one forward pass buys in gradient signal under the two objectives.

Run: python experiments/ch09_counts.py
"""

from __future__ import annotations

import time

from rnn_to_transformer_lab.cost import quadratic_half_point
from rnn_to_transformer_lab.determinism import describe_environment, utf8_stdout
from rnn_to_transformer_lab.scaling import (
    attention_share,
    decoder_only_params,
    encoder_only_params,
    flops_per_token,
    six_nd_error,
)

#: Devlin et al. 2019, section 3: "BERT_BASE (L=12, H=768, A=12, Total
#: Parameters=110M) and BERT_LARGE (L=24, H=1024, A=16, Total
#: Parameters=340M)", with footnote 3 setting the feed-forward size to 4H.
#: The paper says a "30,000 token vocabulary"; the released checkpoints carry
#: 30522, so both are printed below rather than one being chosen.
BERT = (
    ("BERT-base", 12, 768, 3072, 110e6),
    ("BERT-large", 24, 1024, 4096, 340e6),
)
BERT_VOCAB_PAPER = 30000
BERT_VOCAB_RELEASED = 30522
BERT_MAX_POSITIONS = 512

#: Radford et al. 2018, section 4.1: "a 12-layer decoder-only transformer with
#: masked self-attention heads (768 dimensional states and 12 attention
#: heads). For the position-wise feed-forward networks, we used 3072
#: dimensional inner states." Same page: "a bytepair encoding (BPE)
#: vocabulary with 40,000 merges", "contiguous sequences of 512 tokens", and
#: "learned position embeddings instead of the sinusoidal version".
#:
#: **The paper states no parameter count and no vocabulary size.** 40,000 is a
#: merge count, and a BPE vocabulary is the merges plus a base alphabet plus
#: whatever special tokens the implementation adds, so the vocabulary is
#: somewhat larger and by an amount the paper does not give. The commonly
#: quoted "117M" for this model appears nowhere in it. Both facts are why the
#: GPT-1 row below is printed as a range rather than as a number.
GPT1_LAYERS, GPT1_D_MODEL, GPT1_D_FF, GPT1_CTX = 12, 768, 3072, 512
GPT1_MERGES = 40000

#: Radford et al. 2019, table 2, four sizes with a parameter count for each.
#: Section 2.3: "The vocabulary is expanded to 50,257. We also increase the
#: context size from 512 to 1024 tokens".
GPT2 = (
    ("GPT-2 117M", 12, 768, 3072, 117e6),
    ("GPT-2 345M", 24, 1024, 4096, 345e6),
    ("GPT-2 762M", 36, 1280, 5120, 762e6),
    ("GPT-2 1542M", 48, 1600, 6400, 1542e6),
)
GPT2_VOCAB = 50257
GPT2_CTX = 1024

#: Brown et al. 2020, table 2.1. Two rows of that table do not satisfy
#: n_heads * d_head = d_model, which every other row does: GPT-3 XL prints
#: d_model 2048 against 24 heads of 128, and GPT-3 13B prints d_model 5140
#: against 40 heads of 128. Both are transcribed as printed.
GPT3 = (
    ("GPT-3 Small", 12, 768, 12, 64, 125e6),
    ("GPT-3 Medium", 24, 1024, 16, 64, 350e6),
    ("GPT-3 Large", 24, 1536, 16, 96, 760e6),
    ("GPT-3 XL", 24, 2048, 24, 128, 1.3e9),
    ("GPT-3 2.7B", 32, 2560, 32, 80, 2.7e9),
    ("GPT-3 6.7B", 32, 4096, 32, 128, 6.7e9),
    ("GPT-3 13B", 40, 5140, 40, 128, 13.0e9),
    ("GPT-3 175B", 96, 12288, 96, 128, 175.0e9),
)
GPT3_VOCAB = 50257
GPT3_CTX = 2048

#: Devlin et al. 2019, section 3.1.
BERT_MASK_RATE = 0.15


def main() -> None:
    utf8_stdout()
    started = time.perf_counter()
    print(describe_environment())

    print()
    print("1. published parameter counts, rebuilt from published hyperparameters")
    print()
    print("BERT, encoder-only. 'paper' is the figure section 3 prints.")
    print()
    print("model       vocab   pooler mlm   rebuilt        paper      ratio")
    for name, n_layers, d_model, d_ff, printed in BERT:
        for vocab, pooler, mlm in (
            (BERT_VOCAB_PAPER, True, False),
            (BERT_VOCAB_RELEASED, True, False),
            (BERT_VOCAB_RELEASED, True, True),
        ):
            p = encoder_only_params(
                n_layers,
                d_model,
                vocab,
                BERT_MAX_POSITIONS,
                d_ff=d_ff,
                pooler=pooler,
                mlm_head=mlm,
            )
            print(
                f"{name:<11} {vocab:<7} {str(pooler):<6} {str(mlm):<5} "
                f"{p.total:<14,} {printed/1e6:>6.0f}M    {p.total/printed:.4f}"
            )
    print()
    print("GPT-1. The paper gives a merge count, not a vocabulary size, and")
    print("gives no parameter count, so this is a range over plausible")
    print("vocabularies rather than a number.")
    print()
    print("vocabulary   rebuilt")
    for vocab in (GPT1_MERGES, GPT1_MERGES + 256, GPT1_MERGES + 478):
        p = decoder_only_params(
            GPT1_LAYERS, GPT1_D_MODEL, vocab, GPT1_CTX, d_ff=GPT1_D_FF
        )
        note = " (merges only)" if vocab == GPT1_MERGES else ""
        print(f"{vocab:<12} {p.total:<14,}{note}")

    print()
    print("GPT-2, table 2, rebuilt at its own section 2.3 vocabulary and context,")
    print("and then at a vocabulary small enough to reproduce the printed column.")
    print()
    print("model         printed    V=50257 n=1024        gap        gap / d_model")
    for name, n_layers, d_model, d_ff, printed in GPT2:
        p = decoder_only_params(
            n_layers, d_model, GPT2_VOCAB, GPT2_CTX, d_ff=d_ff
        )
        gap = p.total - printed
        print(
            f"{name:<13} {printed/1e6:>6.0f}M   {p.total:<14,} ({p.total/printed:.4f})  "
            f"{gap:>10,.0f}  {gap/d_model:>8.0f}"
        )
    print()
    print("the gap is one embedding table of a constant number of rows, so it is")
    print("a vocabulary difference and not a width or depth error. solving for it:")
    print()
    print("model         V that reproduces the printed count")
    for name, n_layers, d_model, d_ff, printed in GPT2:
        blocks = decoder_only_params(
            n_layers, d_model, 0, GPT2_CTX, d_ff=d_ff
        )
        implied = (printed - blocks.total) / d_model
        print(f"{name:<13} {implied:>10,.0f}")
    print()
    print(f"section 2.3 says the vocabulary was expanded TO {GPT2_VOCAB:,}, so a count")
    print("built from a vocabulary this much smaller is a count taken before that.")
    print()
    print("GPT-3, table 2.1. d_ff is 4*d_model throughout.")
    print()
    print("model         heads*d_head  d_model  rebuilt             paper      ratio")
    for name, n_layers, d_model, n_heads, d_head, printed in GPT3:
        p = decoder_only_params(n_layers, d_model, GPT3_VOCAB, GPT3_CTX)
        flag = "" if n_heads * d_head == d_model else "  <- not d_model"
        print(
            f"{name:<13} {n_heads*d_head:<13} {d_model:<8} {p.total:<19,} "
            f"{printed/1e9:>6.3f}B   {p.total/printed:.4f}{flag}"
        )

    print()
    print("2. the embedding table, as a fraction of the model")
    print("   this is what Kaplan et al. remove before fitting anything")
    print()
    print("model        total               embedding       non-embedding    emb share")
    for label, p in (
        ("GPT-1", decoder_only_params(12, 768, GPT1_MERGES + 478, 512, d_ff=3072)),
        ("GPT-2 1542M", decoder_only_params(48, 1600, GPT2_VOCAB, GPT2_CTX)),
        ("BERT-base", encoder_only_params(12, 768, BERT_VOCAB_RELEASED, 512, d_ff=3072)),
        ("GPT-3 175B", decoder_only_params(96, 12288, GPT3_VOCAB, GPT3_CTX)),
    ):
        print(
            f"{label:<12} {p.total:<19,} {p.embedding:<15,} "
            f"{p.non_embedding:<16,} {p.embedding_fraction:.4f}"
        )

    print()
    print("3. C = 6ND against the exact forward count")
    print("   the ratio is exact_forward / 2N, and D cancels out of it")
    print()
    print("configuration                   n_ctx   ratio    1 + n/(6d)  attn share")
    for label, n_layers, d_model, n_ctx in (
        ("GPT-1        L12 d768", 12, 768, 512),
        ("GPT-2 1542M  L48 d1600", 48, 1600, 1024),
        ("GPT-3 175B   L96 d12288", 96, 12288, 2048),
        ("GPT-3 175B, 4x context", 96, 12288, 8192),
        ("GPT-3 175B, 16x context", 96, 12288, 32768),
    ):
        p = decoder_only_params(n_layers, d_model, GPT3_VOCAB, n_ctx)
        ratio = six_nd_error(n_layers, d_model, n_ctx, p.non_embedding)
        print(
            f"{label:<31} {n_ctx:<7} {ratio:<8.4f} "
            f"{1 + n_ctx/(6*d_model):<11.4f} {attention_share(n_ctx, d_model):.4f}"
        )
    print()
    print("where the dropped term reaches half of a forward pass:")
    print("d_model  n = 6*d_model  cost.quadratic_half_point")
    for d_model in (768, 1600, 12288):
        print(f"{d_model:<8} {6*d_model:<14} {quadratic_half_point(d_model)}")

    print()
    print("4. Kaplan's table 1, term by term, against the exact count")
    print("   per token, one layer, d_attn = d_ff/4 = d_model as section 2.1 sets")
    print()
    print("d_model  n_ctx   kaplan attn   exact attn    ratio  kaplan total  exact total")
    for d_model, n_ctx in ((768, 512), (1600, 1024), (12288, 2048)):
        # Table 1's only n_ctx^2 row: "Attention: Mask", 2 n_layer n_ctx d_attn.
        kaplan_attention = 2 * n_ctx * d_model
        exact = flops_per_token(1, d_model, n_ctx)
        # Table 1's Total: N = 2 d_model n_layer (2 d_attn + d_ff), so the
        # parametric half per token is 2N = 4 d_model (2 d_model + d_ff).
        kaplan_parametric = 4 * d_model * (2 * d_model + 4 * d_model)
        print(
            f"{d_model:<8} {n_ctx:<7} {kaplan_attention:<13,} {exact.attention:<13,} "
            f"{exact.attention/kaplan_attention:<6.2f} "
            f"{kaplan_parametric + kaplan_attention:<13,} {exact.total:,}"
        )
    print()
    print("the parametric halves agree exactly; the attention term does not.")

    print()
    print("5. one forward pass, in gradient signal per token")
    print("   an LM scores every position; masked-LM scores the masked ones")
    print()
    print("objective                targets/token  FLOPs/target, relative")
    print(f"{'left-to-right LM':<24} {1.0:<14.4f} {1.0:.4f}")
    print(
        f"{'masked LM at 15%':<24} {BERT_MASK_RATE:<14.4f} "
        f"{1/BERT_MASK_RATE:.4f}"
    )
    print()
    print("at BERT-base's shape, one pretraining sequence of 512 tokens:")
    seq = 512
    print(f"  positions in the sequence        {seq}")
    print(f"  positions an LM predicts         {seq}")
    print(f"  positions BERT predicts          {int(seq * BERT_MASK_RATE)}")

    print()
    print(f"elapsed {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
