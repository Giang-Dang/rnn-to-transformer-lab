"""Chapter 11: the small ViT's input geometry and matched parameter count."""

from __future__ import annotations

import pytest
import torch

from rnn_to_transformer_lab.vision import SmallCNN, SmallViT, count_parameters


def test_small_vit_has_one_class_token_and_16_image_patches():
    model = SmallViT()
    patches = model.patchify(torch.randn(3, 3, 32, 32))
    assert patches.shape == (3, 16, 192)
    assert model.position.shape == (1, 17, 88)
    assert model(torch.randn(3, 3, 32, 32)).shape == (3, 10)


def test_small_vit_is_parameter_matched_to_the_chapter_10_cnn():
    assert count_parameters(SmallViT()) == 66_095
    assert count_parameters(SmallCNN()) == 66_570


def test_small_vit_rejects_geometry_it_cannot_patch():
    with pytest.raises(ValueError, match="not divisible"):
        SmallViT(image_size=30, patch_size=4)
    with pytest.raises(ValueError, match="expected 32x32"):
        SmallViT().patchify(torch.randn(1, 3, 28, 28))
