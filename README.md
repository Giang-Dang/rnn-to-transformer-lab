# rnn-to-transformer-lab

Companion code for *Từ RNN đến Transformer* (Giang Dang, forthcoming).

Every listing printed in the book comes from this repository at the tag listed
below. Everything here runs on a CPU: no experiment needs a graphics card. That
is a constraint the book set itself, because an experiment that needs hardware
the reader does not have is a figure, not an experiment.

## Getting it running

```
conda env create -f environment.yml
conda activate rnn-to-transformer-lab
python verify.py
```

`verify.py` is the gate. It runs each chapter's verification in order, then the
chapter 3 test suite, then every experiment script, and checks each one
finished inside its time budget. If it prints `verify: ok`, the repo is in the
state the book describes. If it does not, do not trust a number you read here
against the book until it does.

Readers with a graphics card who want to use it: `environment-gpu.yml` is a
separate file, never an edit to the main one, so the default path stays the one
the book promises. The numbers in the book were measured on the CPU
environment, and that is the one to check them against.

## Chapter-to-tag table

| Chapter | Tag | What it ships |
|---------|-----|---------------|
| 01 | `ch01` | Plain RNN forward pass, BPTT with Jacobians, finite-difference gradient check |
| 02 | `ch02` | The adding problem and the copy task, with gradient norm logged per backward step |
| 03 | `ch03` | Jacobian products through time; spectral radius against spectral norm; the error surface of the single-unit model and the wall in it; norm clipping as algorithm 1; the norm-preserving regularizer |
| 04 | `ch04` | The 1997 memory cell with no forget gate; the constant error carousel and its derivative; the input weight conflict solved in closed form; what the paper's truncation costs; parameter counts against the plain layer; the adding problem at lag 100 |
| 05 | `ch05` | The encoder-decoder of Sutskever et al.; a generated English-Vietnamese toy corpus; the fixed-length context vector measured by shrinking it; source reversal; beam search; ensembling |
| 06 | `ch06` | Additive attention as Bahdanau et al. define it, on the same cell and corpus as chapter 5; the bidirectional encoder and a parameter-matched ablation of it; chapter 5's width sweep run again with attention; how far a gradient reaches back into the source in each model; the alignment matrix, and the rate at which it crosses |
| 07 | `ch07` | The Transformer built from the paper's equations: scaled dot-product attention with the variance argument measured, multi-head attention, sinusoidal positional encoding and its linear-shift property, post-LN residual blocks, and the causal mask tested by intervention rather than by reading the code |
| 08 | `ch08` | Table 1 turned into counted FLOPs, bytes and wall clock, with the count checked against torch's own operator instrumentation; the same layer timed against both things "a recurrent layer" can mean; pre-LN as an option on every layer, with the final LayerNorm it needs; the warmup schedule of section 5.3; and label smoothing measured against the loss floor it creates |
| 09 | `ch09` | Closed-form parameter counts for encoder-only and decoder-only stacks, checked against `sum(p.numel())` on modules this repo builds, then used to rebuild the published counts of BERT, GPT-1, GPT-2 and GPT-3 from those papers' own hyperparameter tables; the exact per-token FLOP count of a whole stack against the `C = 6ND` approximation the scaling papers rest on; and the fitted laws of Kaplan et al. and Hoffmann et al. evaluated as functions. Trains nothing and times nothing, so every number here is identical on every machine |
| 10 | `ch10` | A convolution priced as a fully connected layer under two separable constraints, with the published counts of LeCun et al. 1989 and LeNet-5 rebuilt from those papers' own layer tables; translation equivariance measured to bit-exact zero, and its loss under subsampling measured to be total at half of all shifts rather than partial everywhere; a convolution constructed out of `K^2` attention heads, after Cordonnier et al.; and a small convolutional network against parameter-matched and much larger multi-layer perceptrons across training-set sizes. **The first tag that needs data this repo cannot generate** - see below |
| 11 | `ch11` | A parameter-matched Vision Transformer: 8x8 RGB patch embeddings, a learned class token and learned positional embeddings, one post-LN encoder layer, then a linear class head. Its 66,095 parameters sit beside chapter 10's 66,570-parameter CNN; its twelve CIFAR-10 trainings use the exact chapter 10 recipe, while the CNN and MLP reference rows remain the committed chapter 10 canonical run |

Check out a tag to get exactly the code a chapter quotes:

```
git checkout ch03
python verify.py
```

## Checking one chapter

`verify.py` takes `--only`, which runs just that chapter's items:

```
python verify.py --only ch03      # names and budgets: python verify.py --list
```

What a single chapter costs varies more than you might expect. Chapters 1, 3,
4, 7 and 9 are seconds to under two minutes. Chapters 5, 6 and 10 are three to
five, and chapter 8 is six to seven - it trains twenty-four models across two
studies, and the third seed in each is the finding rather than padding.

The whole run is about twenty-three minutes, and it drifts by a couple of
minutes between runs on the same code, because these are wall-clock numbers
from a laptop that is also doing other things. That drift is why the whole-run
total is reported against a target rather than enforced as a budget, while each
individual item's budget is enforced; the note on `TOTAL_TARGET` in `verify.py`
has the reasoning and the arithmetic.

Use `--only` while you are working on one chapter. Run the whole thing before
you trust a number in the book against this repo, which is what `verify: ok`
means.

## From `ch10`, one download

Chapters 1 to 9 build every byte they train on from a seed, so `verify.py` has
never needed a network. Chapter 10's subject is the inductive bias of
convolutional networks and chapter 11's is Vision Transformers, and neither can
be said anything honest about on generated sentences, so from `ch10` this repo
uses CIFAR-10.

Fetch it once. It is 170,498,071 bytes and took about 32 minutes on the machine
the chapter was written on, so it is deliberately not part of the timed work:

```
python -c "from rnn_to_transformer_lab.cifar import fetch; fetch()"
```

It lands in `data/`, which is gitignored, and every later run reads the cache
and needs no network. The archive's MD5 is checked before it is unpacked, so a
truncated download fails with a message rather than with a pickle error. Until
you have fetched once, the chapter 10 CIFAR item fails and tells you this
command.

## Layout

```
rnn_to_transformer_lab/
  __init__.py        chapter 1: the worked three-step RNN, BPTT, gradient check
  ch02_symptoms.py   chapter 2: the adding problem and the copy task
  determinism.py     seeding, and a line naming every version that can move a number
  rnn.py             the plain RNN in the parametrization Pascanu et al. use
  jacobians.py       products of Jacobians through time, and their norms
  surface.py         the single-hidden-unit error surface of the paper's figure 6
  clipping.py        algorithm 1, written out
  regularizer.py     equation 9, the remedy that did not survive
  lstm.py            the 1997 memory cell, the 2000 forget gate, parameter counts
  toy_corpus.py      the generated English-Vietnamese corpus chapters 5 to 8 use
  attention.py       additive attention, the bidirectional encoder, chapter 6
  seq2seq.py         the encoder-decoder, greedy and beam decoding, ensembling
  transformer.py     the Transformer, both layer-norm orders, chapters 7 and 8
  cost.py            table 1 in closed form: FLOPs, crossovers, bytes
  scaling.py         parameter counts and compute budgets, chapter 9: what a
                       published configuration actually holds, and what 6ND
                       leaves out
  conv.py            chapter 10: locality and weight sharing counted separately,
                       the two classic papers' counts rebuilt, and translation
                       equivariance measured exactly rather than asserted
  cifar.py           chapter 10: CIFAR-10, fetched once and cached, MD5 checked
  vision.py          chapter 10: the small CNN and the MLPs it is compared with,
                       and the shared training recipe all of them run
experiments/         one script per set of numbers a chapter prints
  ch08_clock_canonical.txt  the raw run every chapter 8 clock table quotes
  ch08_clock_repeat{2,3,4}.txt  three more independent runs of the same script,
                       which is how chapter 8 shows the head-count slowdown is
                       stable rather than one machine having a bad minute
  ch10_*_canonical.txt  the raw run behind every chapter 10 table. The CIFAR one
                       is machine-dependent and the other two are not; all three
                       are committed so a printed table can be diffed against
                       the run it came from rather than against a memory of it
tests/               those numbers, asserted
verify.py            the gate
```

Two conventions live here at once, and the seam is worth naming rather than
hiding.

Chapters 1 and 2 came first and put one module per chapter, each exposing a
`verify()` that prints its own results. From chapter 3 the package is organized
**by topic instead**, because chapters build on each other: chapter 4 replaces
the recurrence but keeps the Jacobian machinery, and a per-chapter module would
have meant copying it. What pins a chapter's state is its tag, so
`jacobians.py` at `ch03` and the same file at a later tag are allowed to
differ, and the book quotes whichever one its chapter is tagged against.

The chapter number now appears in exactly one place: `experiments/chNN_*.py`.

The chapter 1 and 2 modules are left as they are. Their tags are published, the
book will quote them at those tags, and rewriting them to the new convention
would break that for no reader's benefit.

## Numbers

From chapter 3 on, `tests/` asserts the exact values the book prints, to tight
tolerances. They are deterministic computations on fixed seeds, not training
runs, so a value that moves means something changed.

If a later chapter legitimately changes one of these, change the assertion and
say so in the commit message. Never widen a tolerance to make a run pass.

Reproducibility stops at the PyTorch version: the RNG stream belongs to the
build, not to the seed. `environment.yml` pins what the book measured against,
and every chapter 3 experiment prints the versions actually loaded, so a
mismatch shows up in the output rather than in a wrong assertion.

## Licence

MIT. The papers this code reads are not mine and are not redistributed here;
the book's bibliography names the version of each one it cites.
