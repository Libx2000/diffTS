from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .model import PeriodicConv3d, _groups


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        scale = math.log(10000.0) / max(half - 1, 1)
        frequencies = torch.exp(torch.arange(half, device=t.device) * -scale)
        values = t.float()[:, None] * frequencies[None]
        embedding = torch.cat([values.sin(), values.cos()], dim=1)
        return F.pad(embedding, (0, self.dim - embedding.shape[1]))


class TimeBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, dropout: float):
        super().__init__()
        self.conv1 = PeriodicConv3d(in_channels, out_channels)
        self.norm1 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.conv2 = PeriodicConv3d(out_channels, out_channels)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.time = nn.Linear(time_dim, out_channels * 2)
        self.dropout = nn.Dropout3d(dropout)
        self.skip = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        h = F.silu(self.norm1(self.conv1(x)))
        scale, shift = self.time(temb).chunk(2, dim=1)
        h = h * (1 + scale[:, :, None, None, None]) + shift[:, :, None, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + residual


class ConditionalResidualUNet3D(nn.Module):
    """以历史场和确定性预报为条件，预测扩散噪声。仅下采样水平维度。"""

    def __init__(self, condition_channels: int, state_channels: int = 2, base: int = 16,
                 levels: int = 3, dropout: float = 0.05, time_dim: int = 128):
        super().__init__()
        widths = [base * 2**i for i in range(levels)]
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim), nn.Linear(time_dim, time_dim * 4), nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.encoders = nn.ModuleList()
        current = condition_channels + state_channels
        for width in widths:
            self.encoders.append(TimeBlock(current, width, time_dim, dropout))
            current = width
        self.pool = nn.AvgPool3d((1, 2, 2))
        self.bottleneck = TimeBlock(widths[-1], widths[-1] * 2, time_dim, dropout)
        current = widths[-1] * 2
        self.decoders = nn.ModuleList()
        for width in reversed(widths):
            self.decoders.append(TimeBlock(current + width, width, time_dim, dropout))
            current = width
        self.head = nn.Conv3d(base, state_channels, 1)

    def forward(self, noisy_residual: torch.Tensor, condition: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.time_mlp(t)
        h = torch.cat([noisy_residual, condition], dim=1)
        skips = []
        for block in self.encoders:
            h = block(h, temb)
            skips.append(h)
            h = self.pool(h)
        h = self.bottleneck(h, temb)
        for block, skip in zip(self.decoders, reversed(skips)):
            h = F.interpolate(h, size=skip.shape[-3:], mode="trilinear", align_corners=False)
            h = block(torch.cat([h, skip], dim=1), temb)
        return self.head(h)
