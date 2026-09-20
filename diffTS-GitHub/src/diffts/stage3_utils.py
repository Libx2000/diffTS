from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def spatial_coordinate_channels(artifacts: str | Path, device: torch.device) -> torch.Tensor:
    """返回[1,4,D,H,W]：归一化深度、sin(lat)、sin(lon)、cos(lon)。"""
    coords = np.load(Path(artifacts) / "coordinates.npz")
    depth = np.abs(coords["depth"].astype(np.float32))
    depth = depth / max(float(depth.max()), 1.0)
    lat = np.deg2rad(coords["lat"].astype(np.float32))
    lon = np.deg2rad(coords["lon"].astype(np.float32))
    d, h, w = len(depth), len(lat), len(lon)
    channels = np.stack([
        np.broadcast_to(depth[:, None, None], (d, h, w)),
        np.broadcast_to(np.sin(lat)[None, :, None], (d, h, w)),
        np.broadcast_to(np.sin(lon)[None, None, :], (d, h, w)),
        np.broadcast_to(np.cos(lon)[None, None, :], (d, h, w)),
    ]).astype(np.float32)
    return torch.from_numpy(channels.copy())[None].to(device)
