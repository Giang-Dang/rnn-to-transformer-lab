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
import sys

import numpy as np
import torch


def utf8_stdout() -> None:
    """Print UTF-8 whatever the console claims it can take.

    From chapter 5 the experiments print Vietnamese, and on Windows the default
    console encoding is cp1252, which cannot represent it. `verify.py` captures
    stdout through a pipe, where Python picks the locale encoding rather than
    the console's, so this fails the same way there. Call it before the first
    print in any script whose output is not pure ASCII.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


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
