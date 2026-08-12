"""Chapter 10: what locality and weight sharing each cost, counted exactly.

Nothing here trains or times anything, so every number is reproducible on any
machine to the last digit.

1. One layer under three regimes - dense, local, local and shared - so the two
   constraints that make a convolution can be priced separately.
2. LeCun et al. 1989 rebuilt layer by layer from the paper's own section 3.3.
3. LeNet-5 rebuilt from LeCun et al. 1998 section II.B.

Both rebuilds come back exactly, which is the opposite of what chapter 9 found
for BERT, GPT-2 and GPT-3. Worth saying why the check was run anyway: decision
65 in the book's SPEC says a figure this book prints from a paper's table is
rebuilt from that paper's other tables first, and a rule that only runs when it
is expected to fail is not a rule.

Run: python experiments/ch10_counts.py
"""

from __future__ import annotations

import time

from rnn_to_transformer_lab.conv import layer_counts, lecun89_counts, lenet5_counts
from rnn_to_transformer_lab.determinism import describe_environment, utf8_stdout


def main() -> None:
    utf8_stdout()
    started = time.perf_counter()
    print(describe_environment())

    print()
    print("1. one layer, three regimes")
    print("   32x32x3 in, 32x32x32 out, 3x3 kernel - the first layer of the")
    print("   CIFAR-10 network this chapter trains")
    print()
    print("regime  connections  weights      biases  parameters   conn/param")
    dense, local, conv = layer_counts(32, 32, 3, 32, 32, 32, 3)
    for c in (dense, local, conv):
        print(
            f"{c.regime:<7} {c.connections:<12,} {c.weights:<12,} {c.biases:<7,} "
            f"{c.parameters:<12,} {c.sharing_ratio:>10.2f}"
        )
    print()
    print("each constraint as a divisor on the parameter count")
    print(f"  locality alone      {dense.parameters / local.parameters:>10.2f}")
    print(f"  sharing, on top     {local.parameters / conv.parameters:>10.2f}")
    print(f"  both together       {dense.parameters / conv.parameters:>10.2f}")
    print()
    print("the sharing divisor is the number of spatial positions, 32 x 32.")
    print("locality divides by the ratio of receptive fields; sharing divides")
    print("by however many places the layer looks. they are not equal partners.")

    print()
    print("2. LeCun et al. 1989, rebuilt from section 3.3")
    print("   paper: 1256 units, 64,660 connections, 9,760 parameters")
    print()
    print("layer   connections  weights  biases  parameters  conn/param")
    layers, total = lecun89_counts()
    for c in layers:
        print(
            f"{c.regime:<7} {c.connections:<12,} {c.weights:<8,} {c.biases:<7,} "
            f"{c.parameters:<11,} {c.sharing_ratio:>9.3f}"
        )
    print(
        f"{'TOTAL':<7} {total.connections:<12,} {total.weights:<8,} "
        f"{total.biases:<7,} {total.parameters:<11,} {total.sharing_ratio:>9.3f}"
    )
    print()
    h1 = layers[0]
    print("H1 is the layer worth reading twice. the paper writes its own")
    print("arithmetic out - 768 biases plus 25 times 12 feature kernels -")
    print("and section 3.3 says units do not share their biases, so:")
    print(f"  H1 shared weights            {h1.weights:>7,}")
    print(f"  H1 unshared biases           {h1.biases:>7,}")
    print(f"  biases as a share of H1      {h1.biases / h1.parameters:>7.4f}")
    print()
    shared_part = sum(c.parameters for c in layers[:2])
    dense_part = sum(c.parameters for c in layers[2:])
    print("and where the network's parameters actually sit:")
    print(f"  H1 + H2, the shared layers   {shared_part:>7,}")
    print(f"  H3 + output, fully connected {dense_part:>7,}")
    print(f"  fully connected share        {dense_part / total.parameters:>7.4f}")

    print()
    print("3. LeNet-5, rebuilt from LeCun et al. 1998 section II.B")
    print("   paper: 340,908 connections, 'only 60,000 trainable' parameters")
    print()
    print("layer   connections  parameters")
    layers, total = lenet5_counts()
    for c in layers:
        print(f"{c.regime:<7} {c.connections:<12,} {c.parameters:,}")
    print(f"{'TOTAL':<7} {total.connections:<12,} {total.parameters:,}")
    print()
    print("60,000 is exact rather than rounded. and the connections close")
    print("only once the output layer is counted: 10 RBF units of 84 inputs")
    print("is 840 connections whose weights the paper fixes by hand and")
    print("never trains, so it lands in one total and not the other. leave")
    print("it out and the parameters are still 60,000 while the connections")
    print("are 340,068, exactly 840 short.")
    print()
    conv_part = sum(c.parameters for c in layers[:4])
    dense_part = sum(c.parameters for c in layers[4:])
    print("same reading as 1989, and further along:")
    print(f"  C1..S4, the shared layers    {conv_part:>7,}")
    print(f"  C5 + F6, fully connected     {dense_part:>7,}")
    print(f"  fully connected share        {dense_part / total.parameters:>7.4f}")
    print()
    print("the architecture that introduced weight sharing spends 97% of its")
    print("parameters in the layers that do not share. sharing did not make")
    print("the network small - it made the feature extractor small.")

    print()
    print(f"elapsed {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
