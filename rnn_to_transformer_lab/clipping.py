"""Norm clipping, written out as the paper's algorithm 1.

The whole remedy, transcribed:

    g_hat <- dE/dtheta
    if ||g_hat|| >= threshold then
        g_hat <- (threshold / ||g_hat||) g_hat
    end if

Three things about it are worth keeping in the code rather than only in the
prose. It rescales, so the direction survives and the step stays a descent
direction for the current mini-batch. It fires on a comparison against a fixed
number, so it does nothing at all on the flat part of the surface. And it is
four lines, which is most of why it is the one of the paper's two remedies
still in use.

torch.nn.utils.clip_grad_norm_ is the same algorithm on a parameter list, and
production code should call that. This module exists so chapter 3 can show the
arithmetic and so the experiments can clip a bare tensor without building an
optimizer around it.
"""

from __future__ import annotations

import torch


def clip_norm(gradient: torch.Tensor, threshold: float) -> tuple[torch.Tensor, bool]:
    """Algorithm 1. Returns the possibly rescaled gradient and whether it fired.

    The comparison is >=, matching the paper. At exactly the threshold the
    rescaling multiplies by 1 and changes nothing, so the boundary case is
    cosmetic; it is kept faithful because a reader checking the book against
    the paper should find the same symbol.
    """
    norm = torch.linalg.vector_norm(gradient).item()
    if norm >= threshold:
        return gradient * (threshold / norm), True
    return gradient, False


def clip_parameters(parameters, threshold: float) -> float:
    """What production code should call, kept here so the book can quote it.

    `torch.nn.utils.clip_grad_norm_` is algorithm 1 applied to a whole
    parameter list treated as one vector. It rescales every `.grad` in place
    and returns the norm it saw before clipping. Call it after
    `loss.backward()` and before `optimizer.step()`.

    `clip_norm` above is the same arithmetic on a bare tensor, so that chapter
    3 can show the two lines that matter without building an optimizer around
    them. This function exists so that the one line a reader should actually
    write is real code in this repo rather than a snippet in the prose.
    """
    return torch.nn.utils.clip_grad_norm_(parameters, max_norm=threshold).item()


def step_with_clipping(
    parameters: torch.Tensor,
    gradient: torch.Tensor,
    learning_rate: float,
    threshold: float | None,
) -> tuple[torch.Tensor, bool]:
    """One gradient-descent step, with clipping when `threshold` is not None.

    Returns the new parameters and whether the clip fired. Passing None is the
    unclipped control, and the experiments run both from the same point so the
    only difference between the two trajectories is this argument.
    """
    fired = False
    if threshold is not None:
        gradient, fired = clip_norm(gradient, threshold)
    return parameters - learning_rate * gradient, fired
