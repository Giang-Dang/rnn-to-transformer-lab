# rnn-to-transformer-lab

Companion code for *Từ RNN đến Transformer* (Giang Dang, forthcoming).

Every listing printed in the book comes from this repository at the tag
listed below.  The environment is conda from `environment.yml`; the verify
script activates that env before running anything, so a clean run does not
depend on whose machine it is.

## Chapter-to-tag table

| Chapter | Tag      | What it ships |
|---------|----------|---------------|
| 01      | `ch01`   | Plain RNN forward pass, BPTT with Jacobians, finite-difference gradient check |

## Verify

```
conda env create -f environment.yml
conda activate rnn-to-transformer-lab
python verify.py
```

## Layout

- `rnn_to_transformer_lab/` — one subpackage per chapter group
- `verify.py` — runs every chapter's verification and exits 0
