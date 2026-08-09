"""Seeding, so that a number in the book is the number you get back.

Every experiment in this repo calls `seed_everything` before it allocates a
tensor. The book prints numbers to a few decimals and the verify script
asserts them, which only works if the run is reproducible.

What this does not promise: reproducibility across PyTorch versions. The RNG
stream is a property of the build, not of the seed. `environment.yml` pins the
version the book measured against, and `describe_environment` prints what is
actually loaded so a mismatch is visible in the output rather than in a wrong
assertion.
"""

from __future__ import annotations

import platform
import random

import numpy as np
import torch


def seed_everything(seed: int = 0) -> None:
    """Seed Python, numpy and torch from one number."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def describe_environment() -> str:
    """One line naming everything that can move a measured number."""
    return (
        f"python {platform.python_version()} | "
        f"torch {torch.__version__} | "
        f"numpy {np.__version__} | "
        f"{platform.system()} {platform.machine()}"
    )
