"""Verification script for rnn-to-transformer-lab.

Activate the conda environment before running:

    conda activate rnn-to-transformer-lab
    python verify.py

Each chapter's verification is called in order.  The script exits 0 only
when every check passes.
"""

import sys

from rnn_to_transformer_lab import verify as ch01_verify


def main() -> None:
    print("=== rnn-to-transformer-lab verification ===\n")
    ch01_verify()
    print("\n=== All chapters verified ===")


if __name__ == "__main__":
    main()
