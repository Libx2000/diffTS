from __future__ import annotations

import torch


def make_spatial_weight(lat: torch.Tensor, mask: torch.Tensor, device: torch.device) -> torch.Tensor:
    area = torch.cos(torch.deg2rad(lat.to(device))).clamp_min(1e-6)
    area = area[None, None, None, :, None]
    wet = mask.to(device)[None, None]
    return area * wet


def weighted_mse(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    w = weight.expand(pred.shape[0], pred.shape[1], -1, -1, -1)
    return ((pred - target).square() * w).sum() / w.sum().clamp_min(1.0)


def horizontal_gradient_loss(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    # 经度差分首尾相接；纬度使用相邻差分。
    pred_lon = torch.roll(pred, shifts=-1, dims=-1) - pred
    targ_lon = torch.roll(target, shifts=-1, dims=-1) - target
    pred_lat = pred[..., 1:, :] - pred[..., :-1, :]
    targ_lat = target[..., 1:, :] - target[..., :-1, :]
    w_full = weight.expand(pred.shape[0], pred.shape[1], -1, -1, -1)
    lon = ((pred_lon - targ_lon).abs() * w_full).sum() / w_full.sum().clamp_min(1.0)
    w_lat = w_full[..., :-1, :]
    lat = ((pred_lat - targ_lat).abs() * w_lat).sum() / w_lat.sum().clamp_min(1.0)
    return 0.5 * (lon + lat)

