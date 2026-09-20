from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PeriodicConv3d(nn.Module):
    """经度循环填充；深度和纬度复制填充。"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.pad = pad
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.pad
        x = F.pad(x, (p, p, 0, 0, 0, 0), mode="circular")
        x = F.pad(x, (0, 0, p, p, p, p), mode="replicate")
        return self.conv(x)


def _groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            PeriodicConv3d(in_channels, out_channels),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(),
            nn.Dropout3d(dropout),
            PeriodicConv3d(out_channels, out_channels),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet3D(nn.Module):
    """仅在纬度和经度下采样，15层垂向结构始终保留。"""

    def __init__(self, in_channels: int, out_channels: int = 2, base: int = 24, levels: int = 3, dropout: float = 0.05):
        super().__init__()
        if levels < 2:
            raise ValueError("levels 至少为2")
        widths = [base * (2**i) for i in range(levels)]
        self.encoders = nn.ModuleList()
        current = in_channels
        for width in widths:
            self.encoders.append(ConvBlock(current, width, dropout))
            current = width
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.bottleneck = ConvBlock(widths[-1], widths[-1] * 2, dropout)
        current = widths[-1] * 2
        self.decoders = nn.ModuleList()
        for width in reversed(widths):
            self.decoders.append(ConvBlock(current + width, width, dropout))
            current = width
        self.head = nn.Conv3d(base, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
            x = decoder(torch.cat([x, skip], dim=1))
        return self.head(x)

