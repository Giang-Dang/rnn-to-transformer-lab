"""Chapter 4: the memory cell on the task chapter 2's plain RNN could not do.

Chapter 2 trained a plain recurrent network on the adding problem and watched
it fail: at T = 50 and T = 100 the loss settled around 0.65, which is not a
partial solution but the loss of a model that has stopped listening to its
input. The target is the sum of two values drawn uniformly from [-1, 1], so it
has variance 2/3, and a constant prediction of zero scores 0.6667. Chapter 2's
network scored the variance.

Same task, same generator, same optimizer, same number of sequences. The only
thing that changes is what sits in the recurrence.

The plain RNN here is given the larger hidden layer of the two, so the
comparison cannot be read as the LSTM having been handed more parameters: at
d_h = 32 the plain network carries more weights than the 1997 cell does at
d_h = 16. Both are trained with the same batches in the same order.

Run: python experiments/ch04_adding.py
"""

from __future__ import annotations

import time

import torch

from rnn_to_transformer_lab.determinism import describe_environment, seed_everything
from rnn_to_transformer_lab.lstm import random_lstm

T_LAG = 100
#: Cut from 2500 to fit the book's 60-second per-experiment budget. Nothing is
#: lost by it: the cell is already three orders of magnitude below the
#: predict-zero baseline by batch 800, and the plain RNN is flat from the first
#: batch to the last.
N_BATCHES = 1200
BATCH = 64
LR = 0.01
D_LSTM = 16
D_RNN = 32
SEED = 11
DTYPE = torch.float32
#: Variance of the target: two independent draws from U(-1, 1), each with
#: variance 1/3. A model that always answers zero scores exactly this, and it
#: is the number chapter 2's plain RNN converged to.
TARGET_VARIANCE = 2.0 / 3.0


def adding_batch(
    lag: int, batch: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chapter 2's generator, batched: (lag, batch, 2) inputs and (batch,) targets.

    Column 0 is uniform noise in [-1, 1]; column 1 is zero except at two
    distinct random positions where it is 1.0. The target is the sum of column
    0 at those two positions.
    """
    values = torch.rand(batch, lag, generator=generator, dtype=DTYPE) * 2.0 - 1.0
    marks = torch.zeros(batch, lag, dtype=DTYPE)
    # Two distinct positions per row, drawn without replacement.
    order = torch.rand(batch, lag, generator=generator).argsort(dim=1)[:, :2]
    marks.scatter_(1, order, 1.0)
    targets = (values * marks).sum(dim=1)
    inputs = torch.stack([values, marks], dim=-1).transpose(0, 1).contiguous()
    return inputs, targets


def train_lstm(generator: torch.Generator) -> list[float]:
    gen_w = torch.Generator().manual_seed(SEED)
    model = random_lstm(
        D_LSTM, 2, gen_w, scale=0.1, dtype=DTYPE, requires_grad=True
    )
    readout = (torch.rand(D_LSTM, generator=gen_w, dtype=DTYPE) * 0.2 - 0.1)
    readout.requires_grad_(True)
    bias = torch.zeros(1, dtype=DTYPE, requires_grad=True)
    params = model.parameters() + [readout, bias]
    optimizer = torch.optim.Adam(params, lr=LR)

    losses: list[float] = []
    for step in range(N_BATCHES):
        inputs, targets = adding_batch(T_LAG, BATCH, generator)
        states = model.unroll(inputs, truncate=True)
        prediction = states[-1].h @ readout + bias
        loss = ((prediction - targets) ** 2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def train_rnn(generator: torch.Generator) -> list[float]:
    gen_w = torch.Generator().manual_seed(SEED)

    def block(rows: int, cols: int) -> torch.Tensor:
        t = torch.randn(rows, cols, generator=gen_w, dtype=DTYPE) * 0.1
        return t.requires_grad_(True)

    w_xh = block(D_RNN, 2)
    w_hh = block(D_RNN, D_RNN)
    b_h = torch.zeros(D_RNN, dtype=DTYPE, requires_grad=True)
    readout = block(1, D_RNN)
    bias = torch.zeros(1, dtype=DTYPE, requires_grad=True)
    params = [w_xh, w_hh, b_h, readout, bias]
    optimizer = torch.optim.Adam(params, lr=LR)

    losses: list[float] = []
    for step in range(N_BATCHES):
        inputs, targets = adding_batch(T_LAG, BATCH, generator)
        h = torch.zeros(BATCH, D_RNN, dtype=DTYPE)
        for t in range(T_LAG):
            h = torch.tanh(inputs[t] @ w_xh.T + h @ w_hh.T + b_h)
        prediction = (h @ readout.T).squeeze(-1) + bias
        loss = ((prediction - targets) ** 2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def tail(losses: list[float], n: int = 100) -> float:
    return sum(losses[-n:]) / n


def main() -> None:
    started = time.perf_counter()
    seed_everything(SEED)
    print(describe_environment())
    print(f"adding problem T={T_LAG} batches={N_BATCHES} batch={BATCH} lr={LR}")
    print(f"predict-zero baseline (variance of target) = {TARGET_VARIANCE:.4f}")
    print()

    gen = torch.Generator().manual_seed(SEED)
    lstm_losses = train_lstm(gen)
    lstm_time = time.perf_counter() - started

    gen = torch.Generator().manual_seed(SEED)
    rnn_losses = train_rnn(gen)

    n_lstm = 3 * (D_LSTM * 2 + D_LSTM * D_LSTM + D_LSTM) + D_LSTM + 1
    n_rnn = D_RNN * 2 + D_RNN * D_RNN + D_RNN + D_RNN + 1
    print("model        d_h  params  loss @400    loss @800    final (mean 100)")
    for label, losses, width, count in (
        ("LSTM 1997", lstm_losses, D_LSTM, n_lstm),
        ("plain RNN", rnn_losses, D_RNN, n_rnn),
    ):
        at400 = sum(losses[300:400]) / 100
        at800 = sum(losses[700:800]) / 100
        print(
            f"{label:<12} {width:<4} {count:<7} {at400:<12.6f} "
            f"{at800:<12.6f} {tail(losses):.6f}"
        )

    print()
    print(f"fraction of target variance left unexplained:")
    print(f"  LSTM 1997  {tail(lstm_losses) / TARGET_VARIANCE:.6f}")
    print(f"  plain RNN  {tail(rnn_losses) / TARGET_VARIANCE:.6f}")

    print()
    print(f"lstm training {lstm_time:.1f}s")
    print(f"elapsed {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
