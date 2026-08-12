"""Chapter 9: the two scaling papers' own formulas, evaluated.

Nothing here is fitted and nothing is trained. Every constant is transcribed
from the paper named beside it in `scaling.py`, and this script does the
arithmetic those constants imply - which is the only kind of check available
for a claim about models four orders of magnitude past this repo's budget.

1. Kaplan's three power laws, evaluated, and what they say about where a
   budget should go.
2. Hoffmann's answer to the same question, and the size of the disagreement
   said as a multiplier rather than as an exponent.
3. Hoffmann's own internal consistency: the frontier exponents of table 2
   against the alpha and beta of appendix D.2, which are two halves of the
   same paper that have to agree.
4. Chinchilla against Gopher, and then GPT-3 against a compute-matched model
   the same fit prefers.
5. The "twenty tokens per parameter" rule, against the table it is read from.
6. Whether table 3's own FLOPs column is consistent with C = 6ND.

Run: python experiments/ch09_laws.py
"""

from __future__ import annotations

import time

from rnn_to_transformer_lab.determinism import describe_environment, utf8_stdout
from rnn_to_transformer_lab.scaling import (
    CHINCHILLA_ALPHA,
    CHINCHILLA_BETA,
    KAPLAN_ALPHA_N,
    KAPLAN_P_D,
    KAPLAN_P_N,
    chinchilla_frontier_exponents,
    chinchilla_loss,
    kaplan_compute,
    kaplan_loss_from_compute,
    kaplan_loss_from_params,
    kaplan_loss_from_tokens,
    kaplan_optimal,
)

#: Hoffmann et al. 2022, table 2. Approach, then the fitted a and b with the
#: 10th and 90th bootstrap percentiles the caption defines. The last row is
#: that table's own transcription of Kaplan, printed for comparison.
HOFFMANN_TABLE_2 = (
    ("1. minimum over training curves", 0.50, (0.488, 0.502), 0.50, (0.501, 0.512)),
    ("2. IsoFLOP profiles", 0.49, (0.462, 0.534), 0.51, (0.483, 0.529)),
    ("3. parametric modelling of loss", 0.46, (0.454, 0.455), 0.54, (0.542, 0.543)),
    ("Kaplan et al. 2020", 0.73, None, 0.27, None),
)

#: Table 1: the two models that share a compute budget and split it
#: differently. Parameters, then training tokens.
GOPHER = (280e9, 300e9)
CHINCHILLA = (70e9, 1.4e12)

#: Brown et al. 2020: 175B parameters, 300B tokens (table 2.1's caption).
GPT3 = (175e9, 300e9)

#: Hoffmann et al. 2022, table A4: the label, then num_layers, d_model,
#: ffw_size, num_heads, k/q size, and the ratio of the paper's own exact FLOP
#: count to 6ND. Everything except the ratio is transcribed from the table; the
#: ratio is what section 7 below tries and fails to reproduce.
HOFFMANN_A4 = (
    ("73M", 10, 640, 2560, 10, 64, 1.03),
    ("305M", 20, 1024, 4096, 16, 64, 1.10),
    ("552M", 24, 1280, 5120, 10, 128, 1.08),
    ("1.1B", 26, 1792, 7168, 14, 128, 1.04),
    ("1.6B", 28, 2048, 8192, 16, 128, 1.03),
    ("6.8B", 40, 3584, 14336, 28, 128, 0.99),
)


def hoffmann_exact_forward(n_layers, d_model, ffw, n_heads, kq, n_ctx, vocab):
    """Appendix F's forward-pass count, term by term, in the paper's order.

    Reproduced here rather than imported from `scaling.py` because it is a
    different accounting from this book's: it includes the embedding and
    final-logits matmuls, and it gives the softmax its own line where Kaplan
    calls softmax a sub-leading term and omits it.
    """
    d_attn = n_heads * kq
    per_layer = (
        2 * 3 * n_ctx * d_model * d_attn  # key, query and value projections
        + 2 * n_ctx * n_ctx * d_attn  # key @ query logits
        + 3 * n_heads * n_ctx * n_ctx  # softmax
        + 2 * n_ctx * n_ctx * d_attn  # softmax @ query reductions
        + 2 * n_ctx * d_attn * d_model  # final linear
        + 2 * n_ctx * (2 * d_model * ffw)  # dense block
    )
    return 2 * n_ctx * vocab * d_model + n_layers * per_layer + 2 * n_ctx * d_model * vocab


def hoffmann_params(n_layers, d_model, ffw, n_heads, kq, vocab):
    """Parameters on the paper's own convention: embeddings included."""
    d_attn = n_heads * kq
    per_layer = 3 * d_model * d_attn + d_attn * d_model + 2 * d_model * ffw
    return n_layers * per_layer + vocab * d_model


#: Hoffmann et al. 2022, table 3: parameters, FLOPs, tokens.
HOFFMANN_TABLE_3 = (
    ("400 Million", 400e6, 1.92e19, 8.0e9),
    ("1 Billion", 1e9, 1.21e20, 20.2e9),
    ("10 Billion", 10e9, 1.23e22, 205.1e9),
    ("67 Billion", 67e9, 5.76e23, 1.5e12),
    ("175 Billion", 175e9, 3.85e24, 3.7e12),
    ("280 Billion", 280e9, 9.90e24, 5.9e12),
    ("520 Billion", 520e9, 3.43e25, 11.0e12),
    ("1 Trillion", 1e12, 1.27e26, 21.2e12),
    ("10 Trillion", 10e12, 1.30e28, 216.2e12),
)


def main() -> None:
    utf8_stdout()
    started = time.perf_counter()
    print(describe_environment())

    print()
    print("1. Kaplan's three laws, equations 1.1 to 1.3, evaluated")
    print("   loss in nats per token; each law holds only in its own regime")
    print()
    print("N (non-embedding)   L(N)      D (tokens)      L(D)      C_min (PF-days)  L(C)")
    for n, d, c in ((1e7, 1e8, 1e0), (1e8, 1e9, 1e2), (1e9, 1e10, 1e4), (1e10, 1e11, 1e6)):
        print(
            f"{n:<19.1e} {kaplan_loss_from_params(n):<9.4f} {d:<15.1e} "
            f"{kaplan_loss_from_tokens(d):<9.4f} {c:<16.1e} "
            f"{kaplan_loss_from_compute(c):.4f}"
        )

    print()
    print("2. where an extra factor of compute goes, under each paper")
    print()
    print(f"Kaplan, table 6: N ~ C^{KAPLAN_P_N}, D ~ C^{KAPLAN_P_D}")
    print()
    print("compute x   Kaplan: model x   Kaplan: data x   Hoffmann: each x")
    for factor in (2, 10, 100, 1000):
        kn = factor**KAPLAN_P_N
        kd = factor**KAPLAN_P_D
        h = factor**0.5
        print(f"{factor:<11} {kn:<17.2f} {kd:<16.2f} {h:.2f}")
    print()
    print("Hoffmann's introduction paraphrases the 10x row as '5.5x' and '1.8x'.")
    print(f"Kaplan's own exponents give {10**KAPLAN_P_N:.2f}x and {10**KAPLAN_P_D:.2f}x,")
    print("so that sentence rounds its opponent's numbers rather than quoting them.")

    print()
    print("3. Hoffmann's table 2 against Hoffmann's appendix D.2")
    print()
    print("approach                          a      (10th, 90th)      b      (10th, 90th)")
    for name, a, a_ci, b, b_ci in HOFFMANN_TABLE_2:
        a_s = f"({a_ci[0]}, {a_ci[1]})" if a_ci else "-"
        b_s = f"({b_ci[0]}, {b_ci[1]})" if b_ci else "-"
        print(f"{name:<33} {a:<6.2f} {a_s:<17} {b:<6.2f} {b_s}")
    print()
    derived_a, derived_b = chinchilla_frontier_exponents()
    print("approach 3's row is not an independent fit. equation 4 derives it from")
    print(f"alpha and beta, and appendix D.2 prints those as {CHINCHILLA_ALPHA} and")
    print(f"{CHINCHILLA_BETA}. carrying that through:")
    print(f"  a = beta/(alpha+beta)  = {derived_a:.4f}  rounds to {round(derived_a, 2)}")
    print(f"  b = alpha/(alpha+beta) = {derived_b:.4f}  rounds to {round(derived_b, 2)}")
    print(f"  a + b = {derived_a + derived_b:.4f}  (exactly 1 by construction)")
    print()
    print("table 2 prints 0.46 and 0.54, and 0.4516 does not round to 0.46.")
    print("so the printed alpha and beta are themselves rounded, and the frontier")
    print("was computed before the rounding. what the row implies about them:")
    print()
    print("  beta/alpha must be         ", f"{0.46/0.54:.4f}")
    print(f"  with alpha = {CHINCHILLA_ALPHA}, beta would be  {0.34*0.46/0.54:.4f}")
    print(f"  with beta = {CHINCHILLA_BETA}, alpha would be   {0.28*0.54/0.46:.4f}")
    print("neither printed value survives on its own, so both carry more digits")
    print("than appendix D.2 shows. this is a rounding gap, not a contradiction -")
    print("but it means the loss function and the frontier cannot both be")
    print("reproduced from the constants the paper prints.")

    print()
    print("4. the same compute budget, split two ways")
    print()
    print("model         N            D            C = 6ND      L(N, D)")
    for label, (n, d) in (("Gopher", GOPHER), ("Chinchilla", CHINCHILLA)):
        print(
            f"{label:<13} {n:<12.3g} {d:<12.3g} {kaplan_compute(n, d):<12.3e} "
            f"{chinchilla_loss(n, d):.4f}"
        )
    print()
    ratio = kaplan_compute(*CHINCHILLA) / kaplan_compute(*GOPHER)
    print("section 4 says the two 'have been trained for the same number of FLOPs'.")
    print(f"under 6ND, on the paper's own table 1 figures, they differ by {ratio:.4f}.")
    print()
    print(f"for equal 6ND, Chinchilla at {GOPHER[0]/CHINCHILLA[0]:.0f}x fewer parameters would need")
    print(
        f"{GOPHER[0]*GOPHER[1]/CHINCHILLA[0]:.3e} tokens against the {CHINCHILLA[1]:.1e} it was given."
    )
    print("the paper does not use 6ND for its own accounting - appendix F counts")
    print("every term - and its table A4 reports that exact count runs from 0.99 to")
    print("1.10 times 6ND depending on the shape of the model. Gopher is twice as")
    print("wide as Chinchilla, so the two sit at different points of that spread,")
    print("which is where a 17% gap under 6ND can come from without either figure")
    print("being wrong. this chapter cannot check that, because table A4 gives")
    print("neither the sequence length nor the vocabulary it was computed at.")
    print()
    print("and the same question asked of GPT-3, at GPT-3's own budget:")
    print()
    gpt3_c = kaplan_compute(*GPT3)
    # Under the fit's own ~20 tokens per parameter, C = 6 N (20 N) = 120 N^2.
    matched_n = (gpt3_c / 120.0) ** 0.5
    matched_d = 20.0 * matched_n
    print("model                      N            D            L(N, D)")
    for label, (n, d) in (
        ("GPT-3 as trained", GPT3),
        ("compute-matched, 20 D/N", (matched_n, matched_d)),
    ):
        print(f"{label:<26} {n:<12.4g} {d:<12.4g} {chinchilla_loss(n, d):.4f}")
    print()
    print(f"same C = {gpt3_c:.3e}, and the fit prefers the smaller model by")
    print(
        f"{chinchilla_loss(*GPT3) - chinchilla_loss(matched_n, matched_d):.4f} nats."
    )
    print("table 3's own 175B row wants 3.7e+12 tokens against the 3.0e+11 used,")
    print(f"which is a factor of {3.7e12/300e9:.1f}.")

    print()
    print("5. 'twenty tokens per parameter', against table 3")
    print()
    print("parameters     tokens        D/N     6ND          table's FLOPs  ratio")
    for name, n, flops, d in HOFFMANN_TABLE_3:
        print(
            f"{name:<14} {d:<13.3e} {d/n:<7.1f} {kaplan_compute(n, d):<12.3e} "
            f"{flops:<14.3e} {kaplan_compute(n, d)/flops:.4f}"
        )
    ratios = [d / n for _, n, _, d in HOFFMANN_TABLE_3]
    print()
    print(f"D/N runs {min(ratios):.1f} to {max(ratios):.1f} and drifts upward with scale.")
    print("the paper never writes the rule down; it is a reading of this column.")
    print("the last column is the check that table 3 was built with C = 6ND.")

    print()
    print("6. what one law's exponent buys, said as a percentage")
    print("   L(N) ~ N^-alpha_N, so ten times the model multiplies loss by this")
    print()
    print("factor on N   loss multiplier   reduction")
    for factor in (2, 10, 100):
        mult = factor ** -KAPLAN_ALPHA_N
        print(f"{factor:<13} {mult:<17.4f} {(1-mult)*100:.1f}%")

    print()
    print("7. Hoffmann's table A4, and why this repo cannot reproduce it")
    print("   appendix F counts the embedding and final-logits matmuls, so the")
    print("   ratio depends on sequence length and vocabulary size - and the")
    print("   paper states neither, for the sweep or for table A4.")
    print()
    print("published:   " + "  ".join(f"{r[6]:>5.2f}" for r in HOFFMANN_A4))
    print()
    print("n_ctx  vocab   rebuilt ratios                          worst gap")
    best = None
    for n_ctx in (1024, 2048, 4096):
        for vocab in (32000, 32768, 51200):
            got = [
                hoffmann_exact_forward(L, d, ffw, heads, kq, n_ctx, vocab)
                * 3
                / (6 * hoffmann_params(L, d, ffw, heads, kq, vocab) * n_ctx)
                for _, L, d, ffw, heads, kq, _ in HOFFMANN_A4
            ]
            worst = max(abs(g - r[6]) for g, r in zip(got, HOFFMANN_A4))
            if best is None or worst < best[0]:
                best = (worst, n_ctx, vocab)
            row = "  ".join(f"{g:>5.2f}" for g in got)
            print(f"{n_ctx:<6} {vocab:<7} {row}   {worst:.3f}")
    print()
    print(f"closest of the nine is n_ctx={best[1]}, vocab={best[2]}, and it is still")
    print(f"{best[0]:.3f} off on its worst row. none of the nine reproduces the column.")
    print("recorded as a measurement that was taken and did not work, rather than")
    print("printed beside the paper's column as though the two were comparable.")

    print()
    print("8. Kaplan's own optimum, evaluated at two budgets")
    print()
    print("C_min (PF-days)   N_opt          D_opt          D/N")
    for c in (1e2, 1e4, 1e6):
        n_opt, d_opt = kaplan_optimal(c)
        print(f"{c:<17.1e} {n_opt:<14.4g} {d_opt:<14.4g} {d_opt/n_opt:.2f}")
    print()
    print("the ratio falls as the budget grows, which is the whole disagreement:")
    print("Kaplan's frontier spends a new factor of compute mostly on parameters,")
    print("so tokens per parameter shrinks, and Hoffmann's holds it fixed.")

    print()
    print(f"elapsed {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
