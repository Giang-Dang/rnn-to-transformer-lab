"""Chapter 10: translation equivariance, measured exactly rather than asserted.

Everything here runs at initialization on a fixed seed. Nothing is trained,
because equivariance is a property of the layer's arithmetic and not of any
weights it happens to hold - which is the point, and is why the numbers below
are exact zeros rather than small ones.

1. One convolution, circular padding against zero padding, and where the
   error lives when it is not zero.
2. Subsampling. A stride-2 layer is exactly equivariant to even shifts and not
   equivariant at all to odd ones, and this section measures both halves.
3. Depth. Stack the subsampling and the period multiplies.
4. Cordonnier et al.'s construction: a K x K convolution written as K^2 heads
   each attending to one fixed offset, checked against F.conv2d.

Run: python experiments/ch10_equivariance.py
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn

from rnn_to_transformer_lab.conv import (
    conv_as_attention_residual,
    equivariance_error,
    shift,
)
from rnn_to_transformer_lab.determinism import (
    describe_environment,
    seed_everything,
    utf8_stdout,
)


def main() -> None:
    utf8_stdout()
    started = time.perf_counter()
    print(describe_environment())
    seed_everything(0)

    x = torch.randn(8, 3, 32, 32)

    print()
    print("1. one convolution, 3x3, shifted by one pixel")
    print()
    circular = nn.Conv2d(3, 8, 3, padding=1, padding_mode="circular")
    zero = nn.Conv2d(3, 8, 3, padding=1)
    print("padding     max |f(shift x) - shift f(x)|")
    print(f"circular    {equivariance_error(circular, x, 1, 0):.3e}")
    print(f"zeros       {equivariance_error(zero, x, 1, 0):.3e}")
    print()
    with torch.no_grad():
        d = (zero(shift(x, 1, 0)) - shift(zero(x), 1, 0)).abs()
    print("and for zero padding, where that error sits:")
    print(f"  whole feature map          {d.max().item():.3e}")
    print(f"  interior, borders dropped  {d[:, :, 2:-2, 2:-2].max().item():.3e}")
    print()
    print("the interior is bit-exact. zero padding does not weaken")
    print("equivariance, it breaks it on a two-pixel frame and nowhere else.")

    print()
    print("2. subsampling: conv then MaxPool2d(2)")
    print("   an input shift of 2 should move the output by 1, so that is")
    print("   what the output is compared against")
    print()
    pooled = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1, padding_mode="circular"), nn.MaxPool2d(2)
    )
    print("input shift  output shift  error")
    for k in (0, 2, 4, 6, 8):
        err = equivariance_error(pooled, x, k, 0, downsample=2)
        print(f"{k:<12} {k // 2:<13} {err:.3e}")
    print()
    print("exact at every even shift. now the odd ones, which have no output")
    print("shift to be compared against - so the residual below is the best")
    print("any output shift achieves, searched over all of them:")
    print()
    with torch.no_grad():
        base = pooled(x)
        scale = base.abs().max().item()
        print("input shift  best output shift  residual  residual/scale")
        for k in (1, 3, 5):
            got = pooled(shift(x, k, 0))
            residual, best = min(
                ((got - shift(base, s, 0)).abs().max().item(), s)
                for s in range(-3, 4)
            )
            print(f"{k:<12} {best:<18} {residual:<9.3e} {residual / scale:.4f}")
    print()
    print(f"output magnitude {scale:.4f}, so the best any output shift can")
    print("do for an odd input shift is to miss by more than nine tenths of")
    print("that scale. this is not a degradation that grows with distance -")
    print("the layer has nothing at all to say about half of the shifts.")

    print()
    print("2b. it is the subsampling, not the max")
    print("    Zhang 2019 is usually read as being about max pooling and")
    print("    aliasing. the same test on three ways of halving the grid:")
    print()
    variants = {
        "MaxPool2d(2)": nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1, padding_mode="circular"),
            nn.MaxPool2d(2),
        ),
        "AvgPool2d(2)": nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1, padding_mode="circular"),
            nn.AvgPool2d(2),
        ),
        "Conv stride 2": nn.Conv2d(
            3, 8, 3, stride=2, padding=1, padding_mode="circular"
        ),
    }
    print("layer           even shift  odd shift  scale   odd/scale")
    for name, layer in variants.items():
        even = equivariance_error(layer, x, 2, 0, downsample=2)
        with torch.no_grad():
            base = layer(x)
            got = layer(shift(x, 1, 0))
            odd = min(
                (got - shift(base, s, 0)).abs().max().item() for s in range(-3, 4)
            )
            scale = base.abs().max().item()
        print(f"{name:<15} {even:<11.3e} {odd:<10.3e} {scale:<7.3f} {odd / scale:.3f}")
    print()
    print("all three are bit-exact on even shifts and all three are wrong by")
    print("the order of the signal on odd ones. there is no max operation in")
    print("the third row at all. so the even/odd split is a property of")
    print("throwing away every other sample, not of the nonlinearity that")
    print("happens to sit next to it.")

    print()
    print("3. depth multiplies the period")
    print()
    layers: list[nn.Module] = []
    channels = 3
    print("pool layers  total stride  exact at shifts of")
    for depth in (1, 2, 3):
        layers += [
            nn.Conv2d(channels, 8, 3, padding=1, padding_mode="circular"),
            nn.MaxPool2d(2),
        ]
        channels = 8
        stack = nn.Sequential(*layers)
        stride = 2**depth
        err = equivariance_error(stack, x, stride, 0, downsample=stride)
        assert err == 0.0, err
        print(f"{depth:<12} {stride:<13} {stride}")
    print()
    print("a three-pool network on a 32 pixel image is exact at multiples")
    print("of 8, which is 4 of the 32 horizontal shifts, or one in eight.")
    print("LeCun et al. 1989 said this in words about their own two-pixel")
    print("subsampling: the input image is undersampled and some position")
    print("information is eliminated.")

    print()
    print("4. a convolution written as K^2 attention heads")
    print("   Cordonnier et al. 2020, theorem 1: N_h heads express a")
    print("   sqrt(N_h) x sqrt(N_h) kernel, each head fixed on one offset")
    print()
    print("kernel  heads  max |heads - conv2d|  output scale  relative")
    for k in (3, 5, 7):
        w = torch.randn(8, 3, k, k)
        residual = conv_as_attention_residual(x, w)
        with torch.no_grad():
            padded = nn.functional.pad(x, (k // 2,) * 4, mode="circular")
            scale = nn.functional.conv2d(padded, w).abs().max().item()
        print(
            f"{k}x{k}     {k * k:<6} {residual:<21.3e} {scale:<13.3f} "
            f"{residual / scale:.2e}"
        )
    print()
    print("the residual is float32 accumulation order, not a structural gap:")
    print("it tracks the output scale rather than the kernel size. what the")
    print("construction shows is capacity, not learning - the heads here are")
    print("set by hand, and section 4 of the chapter says so on the page.")

    print()
    print(f"elapsed {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
