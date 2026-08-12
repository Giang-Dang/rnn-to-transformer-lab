"""The two architectures chapter 10 puts side by side on CIFAR-10.

A small convolutional network, and multi-layer perceptrons that see the same
pixels with no locality and no weight sharing. The point of the comparison is
not that the convolutional network wins - it does - but *where* it wins, which
is at the small-data end, and by how much the gap closes as data arrives. That
is the shape chapter 11 needs, because the ViT result is the same curve with a
Transformer on it.

**Two controls rather than one**, per decision 44 in the book's SPEC: an
ablation ships a size-matched control or says why it has none. `mlp_matched` is
built to land within about two percent of the convolutional network's parameter
count, so that a gap between them is a gap about architecture. `mlp_wide` then
gives the MLP more than twenty times the parameters, because a matched MLP
alone is open to the obvious objection - that 66,000 parameters simply buys
more network when they are shared than when they are not, which is exactly what
`experiments/ch10_counts.py` measures - and a reader is owed the version of the
comparison where that objection cannot be raised.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .transformer import EncoderLayer


class SmallCNN(nn.Module):
    """Three 3x3 convolutions with pooling between, then a linear classifier.

    Deliberately plain: no batch norm, no residual, no augmentation. The
    chapter is about what the convolution's two constraints buy, and every
    other modern ingredient is a second explanation for any gap.

    **Zero padding, not circular**, because that is what a real network uses.
    It means the bit-exact equivariance `experiments/ch10_equivariance.py`
    measures does not hold for this model at the border - only in the interior
    - so do not describe this network as exactly equivariant at any shift. Its
    total stride is 8, so the interior features would repeat with period 8 if
    the padding allowed it.
    """

    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width, 2 * width, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(2 * width, 2 * width, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.head = nn.Linear(2 * width * 4 * 4, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x).flatten(1))


class MLP(nn.Module):
    """One hidden layer over the raw 3072 pixels.

    No locality: every hidden unit sees every pixel. No sharing: a feature at
    one corner and the same feature at another are separate parameters, and
    nothing ties them. This is the `dense` row of `layer_counts` made into a
    model.
    """

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallViT(nn.Module):
    """A small Vision Transformer for the chapter 11 CIFAR-10 comparison.

    The architecture keeps only the image-specific choices that ViT itself
    makes: non-overlapping patches, a learned class token, and learned
    positional embeddings.  The encoder layers are the post-LN Transformer
    layers built for chapter 7; the model deliberately has no convolution,
    locality mask, data augmentation, or residual vision-specific component.

    At its defaults the model has 66,095 trainable parameters, 0.9929 times
    `SmallCNN`'s 66,570.  That makes the chapter's comparison about the
    architecture rather than about giving one side more capacity.
    """

    def __init__(
        self,
        *,
        image_size: int = 32,
        patch_size: int = 8,
        channels: int = 3,
        d_model: int = 88,
        n_heads: int = 4,
        n_layers: int = 1,
        d_ff: int = 85,
        n_classes: int = 10,
    ) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError(
                f"image_size {image_size} is not divisible by patch_size {patch_size}"
            )
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_patches = (image_size // patch_size) ** 2
        patch_width = channels * patch_size * patch_size
        self.patch_embedding = nn.Linear(patch_width, d_model)
        self.class_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.position = nn.Parameter(torch.zeros(1, self.n_patches + 1, d_model))
        self.layers = nn.ModuleList(
            EncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Turn NCHW images into a batch of flattened non-overlapping patches."""
        if x.ndim != 4:
            raise ValueError(f"expected NCHW images, got shape {tuple(x.shape)}")
        _, _, height, width = x.shape
        if height != self.image_size or width != self.image_size:
            raise ValueError(
                f"expected {self.image_size}x{self.image_size} images, got {height}x{width}"
            )
        p = self.patch_size
        return (
            x.unfold(2, p, p)
            .unfold(3, p, p)
            .permute(0, 2, 3, 1, 4, 5)
            .reshape(x.shape[0], self.n_patches, -1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embedding(self.patchify(x))
        cls = self.class_token.expand(x.shape[0], -1, -1)
        tokens = torch.cat((cls, patches), dim=1) + self.position
        for layer in self.layers:
            tokens, _ = layer(tokens, None)
        return self.head(self.norm(tokens[:, 0]))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def mlp_matched(target: int) -> MLP:
    """The widest MLP whose parameter count does not exceed `target`.

    3072 inputs means each hidden unit costs 3083 parameters, so the width
    that matches a small convolutional network is small - which is the finding
    rather than an awkwardness of the setup.
    """
    hidden = max(1, (target - 10) // (3 * 32 * 32 + 1 + 10))
    return MLP(hidden)


def standardize(
    train_x: torch.Tensor, *others: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """Center and scale by the *training* split's own per-channel statistics.

    Not by the whole dataset's. A run on 1,000 images that normalizes with all
    50,000 images' mean has been told something about the other 49,000, and the
    whole question this chapter asks is what a model can do with few examples.
    The leak would be small and it would flatter exactly the rows the chapter
    reads most closely.
    """
    mean = train_x.mean(dim=(0, 2, 3), keepdim=True)
    std = train_x.std(dim=(0, 2, 3), keepdim=True)
    return tuple((t - mean) / std for t in (train_x, *others))


def train_and_score(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    *,
    epochs: int,
    batch_size: int = 128,
    lr: float = 1e-3,
) -> tuple[float, float]:
    """Train, then return (test accuracy, train accuracy).

    The training accuracy comes back too because a gap between the two is what
    separates "this model cannot fit the data" from "this model fits it and
    does not generalize", and those are different findings about an inductive
    bias. The same recipe runs for every architecture and every training-set
    size, so a row differs from another row in one thing only.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    n = train_x.shape[0]

    model.train()
    for _ in range(epochs):
        order = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            optimizer.zero_grad()
            loss_fn(model(train_x[idx]), train_y[idx]).backward()
            optimizer.step()

    model.eval()
    # Training accuracy is scored on at most 10,000 examples. It is read only
    # to tell "cannot fit" from "fits and does not generalize", and that
    # question does not need the last decimal; scoring all 50,000 costs real
    # seconds against a budget decision 62 says not to raise.
    limit = min(train_x.shape[0], 10_000)
    return (
        _accuracy(model, test_x, test_y),
        _accuracy(model, train_x[:limit], train_y[:limit]),
    )


def _accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    correct = 0
    with torch.no_grad():
        for start in range(0, x.shape[0], 1000):
            chunk = x[start : start + 1000]
            correct += (model(chunk).argmax(1) == y[start : start + 1000]).sum().item()
    return correct / x.shape[0]
