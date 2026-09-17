"""Residual MinkowskiEngine U-Net for 11-class railway segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from railway3d_seg.labels import NUM_CLASSES


def _dependencies():
    try:
        import MinkowskiEngine as ME
        import torch
        from torch import nn
    except ImportError as error:  # pragma: no cover - depends on local CUDA stack
        raise ImportError(
            "Neural training/inference requires PyTorch and MinkowskiEngine built for the "
            "same CUDA runtime. Evaluation and submission validation remain CPU-only."
        ) from error
    return torch, nn, ME


def build_model(*, in_channels: int = 4, base_channels: int = 32, num_classes: int = NUM_CLASSES):
    """Construct a four-scale sparse residual U-Net lazily."""

    _, nn, ME = _dependencies()

    class ResidualBlock(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.body = nn.Sequential(
                ME.MinkowskiConvolution(channels, channels, kernel_size=3, dimension=3),
                ME.MinkowskiBatchNorm(channels),
                ME.MinkowskiReLU(inplace=True),
                ME.MinkowskiConvolution(channels, channels, kernel_size=3, dimension=3),
                ME.MinkowskiBatchNorm(channels),
            )
            self.activation = ME.MinkowskiReLU(inplace=True)

        def forward(self, features):
            return self.activation(self.body(features) + features)

    def stage(input_channels: int, output_channels: int, *, stride: int):
        return nn.Sequential(
            ME.MinkowskiConvolution(
                input_channels, output_channels, kernel_size=3, stride=stride, dimension=3
            ),
            ME.MinkowskiBatchNorm(output_channels),
            ME.MinkowskiReLU(inplace=True),
            ResidualBlock(output_channels),
            ResidualBlock(output_channels),
        )

    class RailwaySparseUNet(nn.Module):
        def __init__(self):
            super().__init__()
            c = int(base_channels)
            self.stem = stage(in_channels, c, stride=1)
            self.enc2 = stage(c, c * 2, stride=2)
            self.enc3 = stage(c * 2, c * 4, stride=2)
            self.enc4 = stage(c * 4, c * 8, stride=2)
            self.up3 = ME.MinkowskiConvolutionTranspose(
                c * 8, c * 4, kernel_size=2, stride=2, dimension=3
            )
            self.dec3 = stage(c * 8, c * 4, stride=1)
            self.up2 = ME.MinkowskiConvolutionTranspose(
                c * 4, c * 2, kernel_size=2, stride=2, dimension=3
            )
            self.dec2 = stage(c * 4, c * 2, stride=1)
            self.up1 = ME.MinkowskiConvolutionTranspose(
                c * 2, c, kernel_size=2, stride=2, dimension=3
            )
            self.dec1 = stage(c * 2, c, stride=1)
            self.dropout = ME.MinkowskiDropout(p=0.2)
            self.head = ME.MinkowskiConvolution(c, num_classes, kernel_size=1, dimension=3)

        def forward(self, sparse_tensor):
            level1 = self.stem(sparse_tensor)
            level2 = self.enc2(level1)
            level3 = self.enc3(level2)
            level4 = self.enc4(level3)
            decoded3 = self.dec3(ME.cat(self.up3(level4), level3))
            decoded2 = self.dec2(ME.cat(self.up2(decoded3), level2))
            decoded1 = self.dec1(ME.cat(self.up1(decoded2), level1))
            logits = self.head(self.dropout(decoded1))
            query = sparse_tensor.C.to(device=logits.F.device, dtype=logits.F.dtype)
            return logits.features_at_coordinates(query)

    return RailwaySparseUNet()


def load_compatible_weights(model: Any, checkpoint: str | Path) -> tuple[list[str], list[str]]:
    torch, _, _ = _dependencies()
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    state = payload.get("model", payload.get("state_dict", payload))
    current = model.state_dict()
    compatible = {
        key.removeprefix("module."): value
        for key, value in state.items()
        if key.removeprefix("module.") in current
        and current[key.removeprefix("module.")].shape == value.shape
    }
    result = model.load_state_dict(compatible, strict=False)
    return list(result.missing_keys), list(result.unexpected_keys)

