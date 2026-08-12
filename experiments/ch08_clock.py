"""Chapter 8: the same comparison as ch08_flops.py, on a clock instead.

`ch08_flops.py` counts operations, and counting is where the paper's argument
lives. A clock answers a different question, and the two answers do not agree,
which is the whole reason this script exists next to that one.

Three tables.

1. One self-attention sub-layer against two things that are both "a recurrent
   layer": this repo's own Python loop over time steps, and `torch.nn.LSTM`,
   which is the same arithmetic behind one fused kernel. Table 1 of the paper
   says "Recurrent" and means the arithmetic; a reader reaches for the library.
   The gap between those two readings is larger than the gap the table is about.
2. The head sweep chapter 7 owed. Section 3.2.2 says the total computational
   cost of h heads "is similar to that of single-head attention with full
   dimensionality", and `ch08_flops.py` confirms the head count cancels out of
   the FLOP formula exactly. This table is the clock's opinion of that.
3. One row of table 1 repeated with `torch.set_num_threads(1)`, because a
   wall-clock table that does not say how many threads it ran on is not a
   measurement of anything repeatable.

Timing: one warm-up call, then the median of REPEATS runs, with the spread
printed so a row that was disturbed is visible rather than averaged away.

Run: python experiments/ch08_clock.py
"""

from __future__ import annotations

import time

import torch
from torch import nn

from rnn_to_transformer_lab.cost import score_matrix_bytes
from rnn_to_transformer_lab.determinism import (
    describe_environment,
    seed_everything,
    utf8_stdout,
)
from rnn_to_transformer_lab.seq2seq import LstmLayer
from rnn_to_transformer_lab.transformer import MultiHeadAttention

BATCH = 8
REPEATS = 9
WIDTHS = (128, 512)
LENGTHS = (32, 64, 128, 256, 512, 1024)
#: Section 3.2.2's own sweep is over h at fixed d_model; table 3 row (A) runs
#: 1, 4, 16 and 32. These are the powers of two that divide 512.
HEAD_COUNTS = (1, 2, 4, 8, 16, 32)
HEAD_LENGTH = 512
#: Intel Core i7-14700K, 33 MiB of L3. Named rather than inlined because the
#: whole point of table 2 is that this number, and not any n, is the threshold.
L3_BYTES = 33 * 2**20


def timed(call, repeats: int = REPEATS) -> tuple[float, float]:
    """Median seconds over `repeats` runs, and the max-minus-min spread."""
    call()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        call()
        samples.append(time.perf_counter() - started)
    samples.sort()
    return samples[len(samples) // 2], samples[-1] - samples[0]


def attention_forward(layer: MultiHeadAttention, x: torch.Tensor):
    def call() -> None:
        with torch.no_grad():
            layer(x, x, x, None)

    return call


def repo_lstm_forward(layer: LstmLayer, x: torch.Tensor):
    """The repo's own encoder loop: project the whole sequence, then step."""

    def call() -> None:
        with torch.no_grad():
            projected = layer.project(x)
            state = layer.initial_state(x.shape[0], x.dtype)
            for step in range(x.shape[1]):
                state = layer.step_projected(projected[:, step], state)

    return call


def torch_lstm_forward(layer: nn.LSTM, x: torch.Tensor):
    def call() -> None:
        with torch.no_grad():
            layer(x)

    return call


def attention_backward(layer: MultiHeadAttention, x: torch.Tensor):
    def call() -> None:
        layer.zero_grad(set_to_none=True)
        out, _ = layer(x, x, x, None)
        out.sum().backward()

    return call


def repo_lstm_backward(layer: LstmLayer, x: torch.Tensor):
    """The same loop with the graph kept, which is what training pays.

    One autograd node per time step, so the Python loop's cost stops being a
    constant overhead on the forward pass and becomes a graph the backward pass
    has to walk. This is the row that decides how far the repo's own recurrent
    layer can be used as a stand-in for "a recurrent layer" in a cost argument.
    """

    def call() -> None:
        layer.zero_grad(set_to_none=True)
        projected = layer.project(x)
        state = layer.initial_state(x.shape[0], x.dtype)
        for position in range(x.shape[1]):
            state = layer.step_projected(projected[:, position], state)
        state[1].sum().backward()

    return call


def torch_lstm_backward(layer: nn.LSTM, x: torch.Tensor):
    def call() -> None:
        layer.zero_grad(set_to_none=True)
        out, _ = layer(x)
        out.sum().backward()

    return call


def main() -> None:
    utf8_stdout()
    started = time.perf_counter()
    print(describe_environment())
    seed_everything(0)
    default_threads = torch.get_num_threads()

    print()
    print("1. one layer forward, seconds, median of 9")
    print(f"batch {BATCH}, 8 heads, torch threads = {default_threads}")
    print()
    print("d    n     attention   spread      repo loop   nn.LSTM     attn/lstm")
    for d in WIDTHS:
        attention = MultiHeadAttention(d, 8)
        repo = LstmLayer(d, d)
        fused = nn.LSTM(d, d, batch_first=True)
        for n in LENGTHS:
            x = torch.randn(BATCH, n, d)
            a, spread = timed(attention_forward(attention, x))
            r, _ = timed(repo_lstm_forward(repo, x))
            f, _ = timed(torch_lstm_forward(fused, x))
            print(
                f"{d:<4} {n:<5} {a:<11.6f} {spread:<11.6f} {r:<11.6f} "
                f"{f:<11.6f} {a / f:.2f}"
            )

    print()
    print("1b. forward + backward, which is what training actually pays")
    print(f"d = 128, batch {BATCH}, 8 heads, median of 5")
    print("the repo loop builds one autograd node per time step; nn.LSTM does not")
    print()
    print("n     attention   repo loop   nn.LSTM     loop/lstm  attn/lstm")
    attention = MultiHeadAttention(128, 8)
    repo = LstmLayer(128, 128)
    fused = nn.LSTM(128, 128, batch_first=True)
    for n in (128, 256, 512):
        x = torch.randn(BATCH, n, 128, requires_grad=True)
        a, _ = timed(attention_backward(attention, x), repeats=5)
        r, _ = timed(repo_lstm_backward(repo, x), repeats=5)
        f, _ = timed(torch_lstm_backward(fused, x), repeats=5)
        print(
            f"{n:<5} {a:<11.6f} {r:<11.6f} {f:<11.6f} {r / f:<10.1f} {a / f:.2f}"
        )

    print()
    print("2. the same arithmetic, normalised: seconds per gigaFLOP")
    print(f"one attention sub-layer, d = 128, 8 heads, batch {BATCH}")
    print("a cost model that is only about FLOPs predicts this column is flat")
    print(f"this machine's L3 is {L3_BYTES / 2**20:.0f} MiB")
    print()
    print("n     score MiB   gigaFLOP   seconds     s/GFLOP     vs n=64")
    layer = MultiHeadAttention(128, 8)
    baseline_rate = None
    for n in (64, 128, 192, 256, 320, 384, 448, 512, 640):
        x = torch.randn(BATCH, n, 128)
        median, _ = timed(attention_forward(layer, x))
        flops = BATCH * (8 * n * 128 * 128 + 4 * n * n * 128)
        rate = median / (flops / 1e9)
        baseline_rate = rate if baseline_rate is None else baseline_rate
        print(
            f"{n:<5} {score_matrix_bytes(n, 8, BATCH) / 2**20:<11.1f} "
            f"{flops / 1e9:<10.3f} {median:<11.6f} {rate:<11.6f} "
            f"{rate / baseline_rate:.2f}"
        )

    print()
    print("the same bytes reached by holding n and raising the batch instead")
    print("d = 128, 8 heads, n = 256; if the step above is about n it stays put")
    print()
    print("batch  score MiB   gigaFLOP   seconds     s/GFLOP")
    for batch in (2, 4, 8, 16, 32):
        x = torch.randn(batch, 256, 128)
        median, _ = timed(attention_forward(layer, x))
        flops = batch * (8 * 256 * 128 * 128 + 4 * 256 * 256 * 128)
        print(
            f"{batch:<6} {score_matrix_bytes(256, 8, batch) / 2**20:<11.1f} "
            f"{flops / 1e9:<10.3f} {median:<11.6f} {median / (flops / 1e9):.6f}"
        )

    print()
    print("3. the head sweep chapter 7 owed")
    print(f"d_model = 512, batch {BATCH}, forward only")
    print("parameters and FLOPs are identical across every row (see ch08_flops.py)")
    print("swept at two lengths, because one length cannot tell a trend from a fluke")
    print()
    # The parameter count is printed once above the table rather than as a
    # column repeating the same figure six times: that it does not move is the
    # point, and a column of one value states it worse than a sentence does.
    print(f"every row holds {sum(p.numel() for p in MultiHeadAttention(512, 1).parameters()):,} parameters")
    print()
    print("h    d_k   n=128       vs h=1   n=512       spread      vs h=1")
    baselines: dict[int, float] = {}
    for heads in HEAD_COUNTS:
        layer = MultiHeadAttention(512, heads)
        cells = {}
        spread_long = 0.0
        for n in (128, HEAD_LENGTH):
            x = torch.randn(BATCH, n, 512)
            median, spread = timed(attention_forward(layer, x))
            cells[n] = median
            if n == HEAD_LENGTH:
                spread_long = spread
            baselines.setdefault(n, median)
        assert sum(p.numel() for p in layer.parameters()) == 4 * 512 * 512
        print(
            f"{heads:<4} {512 // heads:<5} {cells[128]:<11.6f} "
            f"{cells[128] / baselines[128]:<8.2f} {cells[HEAD_LENGTH]:<11.6f} "
            f"{spread_long:<11.6f} {cells[HEAD_LENGTH] / baselines[HEAD_LENGTH]:.2f}"
        )

    print()
    print("4. the same measurement is a different measurement at one thread")
    print(f"d = 512, batch {BATCH}, forward only")
    print()
    print("threads  n     attention   nn.LSTM     attn/lstm")
    for threads in (default_threads, 1):
        torch.set_num_threads(threads)
        attention = MultiHeadAttention(512, 8)
        fused = nn.LSTM(512, 512, batch_first=True)
        for n in (128, 512):
            x = torch.randn(BATCH, n, 512)
            a, _ = timed(attention_forward(attention, x))
            f, _ = timed(torch_lstm_forward(fused, x))
            print(f"{threads:<8} {n:<5} {a:<11.6f} {f:<11.6f} {a / f:.2f}")
    torch.set_num_threads(default_threads)

    print()
    print(f"elapsed {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
