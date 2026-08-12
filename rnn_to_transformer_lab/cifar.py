"""CIFAR-10, fetched once and cached.

This is the first module in the repo that needs data it cannot generate. Every
experiment from chapter 5 to chapter 9 runs on `toy_corpus.py`, which builds its
own data from a seed, so `verify.py` has never needed a network. Chapter 10's
subject is the inductive bias of convolutional networks and chapter 11's is
Vision Transformers, and neither can be said anything honest about on generated
sentences - so the book's author settled it explicitly: ship the real dataset,
accept the download, and let this module cache it. See decision 69 in the
book's SPEC.

**What that costs, stated plainly.** A reader with no network gets a red
`verify.py` until they have fetched once. That is a real loss and it is the
price of the chapter having evidence at all; the alternative was a chapter that
cites instead of measures, which chapter 9 already had to do once.

**Provenance.** Krizhevsky (2009), "Learning Multiple Layers of Features from
Tiny Images", University of Toronto. 60,000 colour images of 32x32 in 10
classes, 50,000 train and 10,000 test, hand-labeled out of the 80 Million Tiny
Images collection. The archive's MD5 is published on the dataset page and this
module checks it, because a truncated download is otherwise a very confusing
pickle error.

Nothing here reads torchvision. The archive is six pickles in a tarball and a
loader for it is shorter than the dependency would be; `environment.yml` stays
as it was.
"""

from __future__ import annotations

import hashlib
import pickle
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import torch

#: The dataset page at cs.toronto.edu/~kriz/cifar.html redirects to cave.
#: Both the URL and the checksum are printed on that page.
CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_MD5 = "c58f30108f718f92721af3b95e74349a"
CIFAR10_BYTES = 170_498_071

#: Krizhevsky 2009 section 3.1. Order is the order of `batches.meta`.
CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARCHIVE = DATA_DIR / "cifar-10-python.tar.gz"
EXTRACTED = DATA_DIR / "cifar-10-batches-py"


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(*, allow_download: bool = True) -> Path:
    """Make sure the extracted dataset is on disk, downloading if needed.

    Returns the directory holding the six pickles. Raises with an actionable
    message rather than a pickle traceback when there is no data and no
    network.
    """
    if EXTRACTED.is_dir() and (EXTRACTED / "test_batch").is_file():
        return EXTRACTED

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not ARCHIVE.is_file():
        if not allow_download:
            raise RuntimeError(
                f"CIFAR-10 is not cached at {ARCHIVE} and downloading is "
                "disabled. Fetch it once with "
                "`python -c \"from rnn_to_transformer_lab.cifar import fetch; "
                'fetch()"` and every later run works offline.'
            )
        urllib.request.urlretrieve(CIFAR10_URL, ARCHIVE)

    digest = _md5(ARCHIVE)
    if digest != CIFAR10_MD5:
        raise RuntimeError(
            f"{ARCHIVE} has MD5 {digest}, expected {CIFAR10_MD5}. The download "
            "is truncated or the file upstream changed; delete it and retry."
        )

    with tarfile.open(ARCHIVE, "r:gz") as tar:
        tar.extractall(DATA_DIR, filter="data")
    return EXTRACTED


def _load_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    # Unpickling executes arbitrary code, so it is only ever safe on bytes you
    # know the provenance of. These bytes qualify: the archive comes from the
    # URL above over TLS and `fetch` refuses to extract it unless its MD5
    # matches the digest published on the dataset page, which is checked before
    # this function can run. The pickle format is not a choice made here - it
    # is the format Krizhevsky distributes - and re-encoding the dataset to
    # something safer would mean shipping a copy whose provenance is this repo
    # rather than Toronto, which is worse.
    with path.open("rb") as handle:
        batch = pickle.load(handle, encoding="bytes")
    # 10000 x 3072 uint8, stored as all of R then all of G then all of B.
    data = batch[b"data"].reshape(-1, 3, 32, 32)
    labels = np.array(batch[b"labels"], dtype=np.int64)
    return data, labels


def load(*, allow_download: bool = True) -> dict[str, torch.Tensor]:
    """Return the whole dataset as four tensors.

    Images are float32 in [0, 1], NCHW, unnormalized - the experiments
    standardize with the training split's own statistics so that a run on
    1,000 images does not quietly borrow knowledge of all 50,000.
    """
    root = fetch(allow_download=allow_download)

    train_data, train_labels = [], []
    for i in range(1, 6):
        d, l = _load_batch(root / f"data_batch_{i}")
        train_data.append(d)
        train_labels.append(l)
    test_data, test_labels = _load_batch(root / "test_batch")

    return {
        "train_x": torch.from_numpy(np.concatenate(train_data)).float().div_(255.0),
        "train_y": torch.from_numpy(np.concatenate(train_labels)),
        "test_x": torch.from_numpy(test_data).float().div_(255.0),
        "test_y": torch.from_numpy(test_labels),
    }


def describe(data: dict[str, torch.Tensor]) -> None:
    """Print what was actually loaded, so a wrong cache is visible.

    The per-class counts are printed as a range rather than as ten numbers,
    because the fact worth seeing is that they are all the same: Krizhevsky
    2009 says the split gives every class exactly 5,000 training and 1,000 test
    images, so a min that differs from a max means the wrong bytes are on disk.
    """
    for split in ("train", "test"):
        x, y = data[f"{split}_x"], data[f"{split}_y"]
        counts = torch.bincount(y, minlength=10)
        lo, hi = counts.min().item(), counts.max().item()
        per_class = f"{lo:,} per class" if lo == hi else f"{lo:,}..{hi:,} per class"
        print(f"{split:<5} {tuple(x.shape)}  {y.shape[0]:,} labels, {per_class}")
    print(f"classes {len(CLASSES)}: {CLASSES[0]} .. {CLASSES[-1]}")
