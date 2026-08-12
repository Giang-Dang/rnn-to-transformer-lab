"""Chapter 11: a parameter-matched ViT on CIFAR-10.

The chapter 10 sweep stays untouched because its canonical output belongs to
that chapter. This repeats its data split, standardisation, recipe and seeds,
then trains its small Vision Transformer. The CNN and MLP rows are the
committed `ch10` canonical run: their code and recipe are unchanged.
Raw output is committed as ch11_vit_canonical.txt after this script passes.

Run: python -m experiments.ch11_vit
"""

from __future__ import annotations

import time

import torch

from rnn_to_transformer_lab.cifar import describe, load
from rnn_to_transformer_lab.determinism import (
    describe_environment,
    seed_everything,
    utf8_stdout,
)
from rnn_to_transformer_lab.vision import (
    MLP,
    SmallCNN,
    SmallViT,
    count_parameters,
    mlp_matched,
    standardize,
    train_and_score,
)

# This is the chapter 10 protocol verbatim. Keeping it here rather than
# importing the experiment makes the chapter's executable evidence independent
# of a script whose output and canonical run belong to another chapter.
SWEEP = ((1000, 25), (4000, 14), (16000, 7), (50000, 4))
SEEDS = (0, 1, 2)
WIDE_HIDDEN = 512


def make_model(name: str, cnn_params: int):
    if name == "CNN":
        return SmallCNN()
    if name == "ViT matched":
        return SmallViT()
    if name == "MLP matched":
        return mlp_matched(cnn_params)
    if name == "MLP wide":
        return MLP(WIDE_HIDDEN)
    raise ValueError(f"unknown model {name}")


def main() -> None:
    utf8_stdout()
    started = time.perf_counter()
    print(describe_environment())

    data = load()
    print()
    print("CIFAR-10, Krizhevsky 2009")
    describe(data)

    train_x, test_x = data["train_x"], data["test_x"]
    train_y, test_y = data["train_y"], data["test_y"]
    cnn_params = count_parameters(SmallCNN())
    names = ("CNN", "ViT matched", "MLP matched", "MLP wide")

    print()
    print("1. four models, before training")
    print()
    print("model         parameters  vs CNN  locality  sharing")
    for name in names:
        params = count_parameters(make_model(name, cnn_params))
        locality = "yes" if name == "CNN" else "no"
        sharing = "yes" if name in ("CNN", "ViT matched") else "no"
        print(
            f"{name:<13} {params:<11,} {params / cnn_params:<7.4f} "
            f"{locality:<9} {sharing}"
        )
    print()
    print("ViT patches are 8x8 RGB pixels: 16 image tokens plus one class token.")
    print("It has learned positions, but no local-attention mask or convolution.")

    print()
    print("2. ViT test accuracy against training-set size")
    print(f"   {len(SEEDS)} seeds, full 10,000-image test split, chapter 10 recipe")
    print()
    print("n_train  epochs  model         test acc     spread   train acc")

    results: dict[tuple[str, int], list[float]] = {}
    train_results: dict[tuple[str, int], list[float]] = {}
    for n_train, epochs in SWEEP:
        for name in ("ViT matched",):
            test_scores, train_scores = [], []
            for seed in SEEDS:
                seed_everything(seed)
                model = make_model(name, cnn_params)
                subset = torch.randperm(train_x.shape[0])[:n_train]
                sub_x, sub_test_x = standardize(train_x[subset], test_x)
                test_acc, train_acc = train_and_score(
                    model,
                    sub_x,
                    train_y[subset],
                    sub_test_x,
                    test_y,
                    epochs=epochs,
                )
                test_scores.append(test_acc)
                train_scores.append(train_acc)
            results[(name, n_train)] = test_scores
            train_results[(name, n_train)] = train_scores
            mean = sum(test_scores) / len(test_scores)
            spread = max(test_scores) - min(test_scores)
            train_mean = sum(train_scores) / len(train_scores)
            print(
                f"{n_train:<8} {epochs:<7} {name:<13} {mean:<12.4f} "
                f"{spread:<8.4f} {train_mean:.4f}"
            )
        print()

    print("The CNN and MLP comparison rows are in ch10_cifar_canonical.txt.")
    print("This output becomes canonical evidence only after this run is committed.")
    print(f"elapsed {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
